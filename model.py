import os
import torch
import time
from typing import List, Tuple, Iterable
import random
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from pydantic import BaseModel

torch.manual_seed(42)
# to rebuild the transfomer from scratch again we need to do the following few things:

# 1. Data Processing
#   We need to create a tokenizer which can read in the dataset & convert
#   it into a consistent format
#   - Tokenizer
#   - Data Processing
#   - Create Dataset
#   - Batch Creation (DataLoadder)
#   -
# 2. The transformer itself
#   For a multi-head attention transformer, we'd have the following variables dictating size:
#     1. Vocab size
#     2. Num embeddings (C)
#     3. Context length (T)
#
#   The architecture of the transformer itself is fairly straightforward:
#   Head
#     1. Token embeddings
#     2. Position embeddings
#   Transformer block
#     1. LayerNorm
#     2. Self-attention
#     3. LayerNorm
#     4. Multi-layer perceptron
#   Tail
#     1. LMHead (linear layer, maps input to the vocab dimesnion, allows us to have a probability distribution)
#     2. Softmax


class AdamWOptimizer:
  def __init__(
    self,
    params: Iterable[nn.Parameter],
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
  ):
    self.params = [p for p in params]
    self.mt = [torch.zeros_like(p) for p in self.params]
    self.vt = [torch.zeros_like(p) for p in self.params]
    self.b1_accum = 1
    self.b2_accum = 1
    self.lr = lr
    self.b1 = betas[0]
    self.b2 = betas[1]
    self.decay = weight_decay
    self.eps = eps

  def zero_grad(self, set_to_none=True):
    for p in self.params:
      p.grad = None

  @torch.no_grad()
  def step(self):
    self.b1_accum *= self.b1
    self.b2_accum *= self.b2
    for i, p in enumerate(self.params):
      # implement just adam for now, we will add the weight decay afterwards
      self.mt[i] = self.b1 * self.mt[i] + (1 - self.b1) * p.grad.data
      self.vt[i] = self.b2 * self.vt[i] + (1 - self.b2) * (p.grad.data**2)

      # corrections
      mt_corrected = self.mt[i] / (1 - self.b1_accum)
      vt_corrected = self.vt[i] / (1 - self.b2_accum)

      # update rule
      p.data -= self.lr * (mt_corrected / (torch.sqrt(vt_corrected) + self.eps))


class Tokenizer:
  def __init__(self, ds: str):
    # build a vocab
    self.vocab = sorted(set(ds))
    self.vocab_size = len(self.vocab)
    self.stoi = {c: i for i, c in enumerate(self.vocab)}
    self.itos = {i: c for c, i in self.stoi.items()}

  def encode(self, s: str, return_tensors=False):
    contents = [self.stoi[c] for c in s]
    if return_tensors:
      contents = torch.tensor(contents, dtype=torch.long)
    return contents

  def decode(self, s: List[int] | torch.Tensor) -> str:
    if isinstance(s, torch.Tensor):
      if s.dim() > 1:
        raise ValueError("cannot handle dimensions greater than 1")
      s = s.tolist()
    return "".join([self.itos[idx] for idx in s])


class Dataset:
  def __init__(self, data: torch.Tensor, context_size: int):
    self.data = data
    self.context_size = context_size

  def __len__(self):
    return max(len(self.data) - (self.context_size + 1), 0)

  def __getitem__(self, idx: int):
    x = self.data[idx : idx + self.context_size]
    y = self.data[idx + 1 : idx + 1 + self.context_size]
    return x, y


def create_dataset(
  text_data: str, tokenizer: Tokenizer, context_size: int
) -> [Dataset, Dataset]:
  processed_data = tokenizer.encode(text_data, return_tensors=False)
  n = int(len(text_data) * 0.9)
  train_data = torch.tensor(processed_data[:n], dtype=torch.long)
  test_data = torch.tensor(processed_data[n:], dtype=torch.long)
  train_ds = Dataset(train_data, context_size=context_size)
  test_ds = Dataset(test_data, context_size=context_size)
  return train_ds, test_ds


class ModelConfig(BaseModel):
  embedding_dim: int
  vocab_size: int
  context_size: int
  num_layers: int
  num_heads: int
  dropout: float


class AttentionHead(nn.Module):
  def __init__(
    self, num_embeddings: int, head_size: int, context_size: int, dropout: float
  ):
    super().__init__()
    self.query = nn.Linear(num_embeddings, head_size)
    self.key = nn.Linear(num_embeddings, head_size)
    self.value = nn.Linear(num_embeddings, head_size)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

  def forward(self, input: torch.Tensor):
    B, T, C = input.shape
    q = self.query(input)  # (B, T, C/n)
    k = self.key(input)  # (B, T, C/n)
    wei = q @ k.transpose(-2, -1) * (C**-0.5)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)  # (B, T, T)

    v = self.value(input)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(
    self,
    num_heads: int,
    head_size: int,
    num_embeddings: int,
    dropout: float,
    context_size: int,
  ):
    super().__init__()
    self.heads = nn.ModuleList(
      [
        AttentionHead(num_embeddings, head_size, context_size, dropout)
        for _ in range(num_heads)
      ]
    )
    self.dropout = nn.Dropout(dropout)
    self.proj = nn.Linear(num_embeddings, num_embeddings)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.cat([h(x) for h in self.heads], dim=-1)
    x = self.proj(x)
    x = self.dropout(x)
    return x


class TransformerBlock(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    # we need a few components:
    self.ln1 = nn.LayerNorm(config.embedding_dim)
    self.ln2 = nn.LayerNorm(config.embedding_dim)

    head_size = config.embedding_dim // config.num_heads
    self.sa = MultiHeadAttention(
      config.num_heads,
      head_size,
      config.embedding_dim,
      config.dropout,
      config.context_size,
    )
    self.mlp = nn.Sequential(
      nn.Linear(config.embedding_dim, config.embedding_dim * 4),
      nn.ReLU(),
      nn.Linear(config.embedding_dim * 4, config.embedding_dim),
      nn.Dropout(config.dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.sa(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.token_embeddings = nn.Embedding(
      num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
    )
    self.pos_embeddings = nn.Embedding(
      num_embeddings=config.context_size,
      embedding_dim=config.embedding_dim,
    )
    self.layers = nn.Sequential(
      *[TransformerBlock(config) for _ in range(config.num_layers)]
    )
    self.ln_f = nn.LayerNorm(config.embedding_dim)
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)
    self.vocab_size = config.vocab_size

  def forward(
    self, inputs: torch.Tensor, targets: torch.Tensor | None = None
  ) -> Tuple[torch.Tensor, torch.Tensor | None]:
    B, T = inputs.shape
    # inputs: (B, T)
    tok_emb = self.token_embeddings(inputs)  # (B, T, C)
    pos_emb = self.pos_embeddings(torch.arange(T))  # (B, T, C)
    x = tok_emb + pos_emb
    x = self.layers(x)
    x = self.ln_f(x)
    x = self.lm_head(x)  # (B, T, V)

    logits = F.softmax(x, dim=-1)

    if targets is None:
      return logits, None

    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss


def train(model: Transformer, optimizer: AdamWOptimizer, train_data: Dataset):
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
  # test_loader = DataLoader(
  #   test_ds, batch_size=script_args.batch_size, shuffle=True, num_workers=4
  # )

  epoch = 0
  max_epochs = 1
  time_running_avg = 0
  time_decay = 0.99
  time_decay_accum = 1
  batch_size = 32
  samples_seen = 0
  step = 0
  save_frequency = 1000

  while epoch < max_epochs:
    for batch in train_loader:
      inputs, targets = batch
      # inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
      t1 = time.time()

      optimizer.zero_grad(set_to_none=True)

      _, loss = model(inputs, targets)
      loss.backward()
      optimizer.step()

      t2 = time.time()
      time_delta = t2 - t1
      time_decay_accum *= time_decay
      time_running_avg = time_decay * time_running_avg + (1 - time_decay) * time_delta
      corrected_running_avg = time_running_avg / (1 - time_decay_accum)

      samples_seen += batch_size
      step += 1
      if step % 100 == 0:
        # calculate average here
        approx_time_left = corrected_running_avg * (len(train_data) - samples_seen)
        print(f"DATASET LENGTH: {len(train_data)}")
        print(f"SAMPLES SEEN: {samples_seen}")
        print(f"DIFF: {len(train_data) - samples_seen}")
        print(
          f"[{epoch}/{max_epochs}] step: {step} samples seen: {samples_seen} loss: {loss.item()} mean time/step: {corrected_running_avg} time left approx: {approx_time_left:.2f} secs | {approx_time_left/60:.2f} mins {approx_time_left/(60**2):.2f} hours | running avg/step: {corrected_running_avg:.2f}secs"
        )
      # if step % EVAL_FREQ == 0:
      #   eval_loss = eval_model(
      #     model,
      #     test_loader=test_loader,
      #   )
      #   train_loss, test_loss = loss.item(), eval_loss.item()
      #   print(
      #     f"[{epoch}/{script_args.num_epochs}] samples seen: {samples_seen}/{len(train_ds)} train loss: {train_loss}, eval loss: {test_loss}"
      #   )

      if step % save_frequency == 0:
        # save_checkpoint(model, script_args.model_dir)
        print("should save")

      # torch.cuda.empty_cache()
    epoch += 1
  return model


def main():
  dist.init_process_group("nccl")
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  with open("tiny-shakespeare.txt", "r") as infile:
    contents = infile.read()

  tokenizer = Tokenizer(contents)

  model_config = ModelConfig(
    context_size=256,
    embedding_dim=384,
    num_heads=6,
    num_layers=8,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1,
  )
  train_data, test_data = create_dataset(contents, tokenizer, model_config.context_size)
  tformer = Transformer(model_config)

  total_params = sum(p.numel() for p in tformer.parameters())
  print("\033[92mInitialized transformer model\033[0m")
  print(f"\033[92mTotal params: {total_params:,}\033[0m")
  optimizer = AdamWOptimizer(tformer.parameters())
  tformer.train()
  train(tformer, optimizer, train_data)
  inputs = torch.randint(0, tokenizer.vocab_size, (32,)).view(1, -1)
  targets = torch.randint(0, tokenizer.vocab_size, (32,)).view(1, -1)
  outputs, loss = tformer(inputs, targets)
  print(outputs)
  print(loss)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()

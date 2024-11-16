import argparse
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

    # this is where we may have some problems
    device = torch.cuda.current_device()
    pos_emb = self.pos_embeddings(torch.arange(T, device=device))  # (B, T, C)

    x = tok_emb + pos_emb
    x = self.layers(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, V)

    if targets is None:
      return logits, None

    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss

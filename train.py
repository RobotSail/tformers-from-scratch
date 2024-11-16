import torch.distributed as dist
import os
import time
import argparse
import torch

from torch.optim import AdamW

from torch.utils.data import DataLoader

from model import (
  Tokenizer,
  Transformer,
  AdamWOptimizer,
  ModelConfig,
  create_dataset,
  Dataset,
)

from pydantic import BaseModel


class ScriptArgs(BaseModel):
  num_epochs: int
  batch_size: int
  save_frequency: int
  save_at_epoch: bool


def train(
  model: Transformer, optimizer: AdamWOptimizer, train_data: Dataset, args: ScriptArgs
):
  model.train()
  train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
  )
  device = torch.cuda.current_device()
  local_rank = int(os.environ["LOCAL_RANK"])
  # test_loader = DataLoader(
  #   test_ds, batch_size=script_args.batch_size, shuffle=True, num_workers=4
  # )

  epoch = 0
  time_running_avg = 0
  time_decay = 0.99
  time_decay_accum = 1
  samples_seen = 0
  step = 0

  while epoch < args.num_epochs:
    for batch in train_loader:
      inputs, targets = batch
      inputs, targets = inputs.to(device), targets.to(device)
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

      samples_seen += inputs.shape[0]
      step += 1
      if local_rank == 0 and step % 100 == 0:
        # calculate average here
        approx_time_left = corrected_running_avg * (len(train_data) - samples_seen)
        print(f"DATASET LENGTH: {len(train_data)}")
        print(f"SAMPLES SEEN: {samples_seen}")
        print(f"DIFF: {len(train_data) - samples_seen}")
        print(
          f"[{epoch}/{args.num_epochs}] step: {step} samples seen: {samples_seen} loss: {loss.item()} mean time/step: {corrected_running_avg} time left approx: {approx_time_left:.2f} secs | {approx_time_left/60:.2f} mins {approx_time_left/(60**2):.2f} hours | running avg/step: {corrected_running_avg:.2f}secs"
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

      if local_rank == 0 and step % args.save_frequency == 0:
        # save_checkpoint(model, script_args.model_dir)
        print("should save")

      torch.cuda.empty_cache()
    epoch += 1
    if local_rank == 0 and args.save_at_epoch:
      print("\033[92m" + f"saving at epoch, samples seen: {samples_seen}" + "\033[0m")
  return model


def main():
  # init distributed
  dist.init_process_group("nccl")
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  torch.cuda.set_device(local_rank)
  device = torch.cuda.current_device()
  print(f"current device is: {device=}")

  # ----------------------------------------
  #           argument parser
  # ----------------------------------------
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--num-epochs", type=int, default=1)
  parser.add_argument("--save-frequency", type=int, default=1000)
  parser.add_argument("--save-at-epoch", type=bool, default=False)
  args = parser.parse_args()

  script_args = ScriptArgs(
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    save_at_epoch=args.save_at_epoch,
    save_frequency=args.save_frequency,
  )

  # ----------------------------------------
  #           init dataset & tokenizer
  # ----------------------------------------
  with open("tiny-shakespeare.txt", "r") as infile:
    contents = infile.read()

  tokenizer = Tokenizer(contents)

  # ----------------------------------------
  #           init model
  # ----------------------------------------
  model_config = ModelConfig(
    context_size=256,
    embedding_dim=384,
    num_heads=6,
    num_layers=8,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1,
  )
  tformer = Transformer(model_config)
  tformer.to(device)

  total_params = sum(p.numel() for p in tformer.parameters())
  print("\033[92mInitialized transformer model\033[0m")
  print(f"\033[92mTotal params: {total_params:,}\033[0m")

  # optimizer = AdamWOptimizer(tformer.parameters())
  optimizer = AdamW(tformer.parameters())

  # ----------------------------------------
  #           init data
  # ----------------------------------------
  train_data, test_data = create_dataset(contents, tokenizer, model_config.context_size)

  # ----------------------------------------
  #           train the model
  # ----------------------------------------
  train(tformer, optimizer, train_data, script_args)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()

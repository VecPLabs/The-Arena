"""
Training script for Grok-vecp-322m (322M parameter Transformer).

Model: 24-layer Transformer, d_model=1024, GQA 16q/4kv, SwiGLU ffn_dim=2816,
       RoPE theta=10000, RMSNorm, weight-tied embeddings.
Dataset: HuggingFaceFW/fineweb-edu (streaming, full train split).
Tokenizer: GPT-2 (vocab 50257).
"""

import os
import sys
import time
import math
import random
import importlib

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Import model from hyphenated filename
# ---------------------------------------------------------------------------
mod = importlib.import_module("grok-vecp-322m_fixed")
ModelConfig = mod.ModelConfig
TransformerModel = mod.TransformerModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
SEQ_LEN = 4096
MICRO_BATCH = 4
GRAD_ACCUM = 32
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM  # 128
PEAK_LR = 6e-4
MIN_LR = 1e-5
WARMUP_STEPS = 750
TOTAL_STEPS = 7500
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
WEIGHT_DECAY = 0.1
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
CHECKPOINT_DIR = "./checkpoints/grok/"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * (step / WARMUP_STEPS)
    if step >= TOTAL_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)


# ---------------------------------------------------------------------------
# Streaming packed-sequence dataset
# ---------------------------------------------------------------------------
class PackedStreamingDataset(IterableDataset):
    """
    Streams fineweb-edu, tokenizes on the fly, concatenates all tokens
    with EOS separators between documents, and yields chunks of exactly
    SEQ_LEN tokens. Remainder tokens are dropped.
    """

    def __init__(self, tokenizer, seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        buffer = []
        chunk_len = self.seq_len + 1  # +1 so we can split into input/target
        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue
            token_ids = self.tokenizer.encode(text)
            buffer.extend(token_ids)
            buffer.append(self.eos_id)
            while len(buffer) >= chunk_len:
                yield torch.tensor(buffer[:chunk_len], dtype=torch.long)
                buffer = buffer[chunk_len :]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert tokenizer.vocab_size == 50257, f"Unexpected vocab size: {tokenizer.vocab_size}"

    # Model
    config = ModelConfig()
    model = TransformerModel(config).to(device)
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters (unique, weight-tied): {unique_params:,}")

    # Try torch.compile
    try:
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: unavailable ({e})")

    # Optimizer — apply weight decay to all params except biases, norms, embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=(BETA1, BETA2),
        eps=EPS,
    )

    # Dataset / DataLoader
    dataset = PackedStreamingDataset(tokenizer, seq_len=SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.train()
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    tokens_total = 0
    t_start = time.time()
    t_log = time.time()
    tokens_at_log = 0

    print(f"\n{'='*60}")
    print(f"Training Grok-vecp-322m")
    print(f"  Total steps:      {TOTAL_STEPS}")
    print(f"  Warmup steps:     {WARMUP_STEPS}")
    print(f"  Micro batch:      {MICRO_BATCH}")
    print(f"  Grad accum:       {GRAD_ACCUM}")
    print(f"  Effective batch:  {EFFECTIVE_BATCH}")
    print(f"  Sequence length:  {SEQ_LEN}")
    print(f"  Peak LR:          {PEAK_LR}")
    print(f"  Min LR:           {MIN_LR}")
    print(f"  Precision:        bf16 mixed")
    print(f"{'='*60}\n")

    data_iter = iter(dataloader)

    while global_step < TOTAL_STEPS:
        # Set LR for this step
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for accum_idx in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the stream if exhausted (unlikely for fineweb-edu full)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch[:, :-1].to(device)    # (B, SEQ_LEN)
            targets = batch[:, 1:].to(device)        # (B, SEQ_LEN)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)             # (B, SEQ_LEN-1, vocab)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                loss = loss / GRAD_ACCUM

            loss.backward()
            step_loss += loss.item()
            tokens_total += input_ids.numel()
            micro_step += 1

        # No gradient clipping (per design doc)
        optimizer.step()
        global_step += 1

        accum_loss += step_loss

        # Logging
        if global_step % LOG_INTERVAL == 0:
            avg_loss = accum_loss / LOG_INTERVAL
            now = time.time()
            elapsed = now - t_log
            toks_delta = tokens_total - tokens_at_log
            throughput = toks_delta / elapsed if elapsed > 0 else 0.0
            total_elapsed = now - t_start
            print(
                f"step {global_step:>6d}/{TOTAL_STEPS} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tokens {tokens_total:,} | "
                f"tok/s {throughput:,.0f} | "
                f"elapsed {total_elapsed:.1f}s"
            )
            accum_loss = 0.0
            t_log = now
            tokens_at_log = tokens_total

        # Checkpoint
        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}.pt")
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "tokens_total": tokens_total,
                },
                ckpt_path,
            )
            print(f"  -> Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"step_{TOTAL_STEPS}.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "tokens_total": tokens_total,
        },
        final_path,
    )
    print(f"  -> Final checkpoint saved: {final_path}")

    total_time = time.time() - t_start
    hours = total_time / 3600
    print(f"\nTraining complete. Total time: {total_time:.1f}s ({hours:.2f}h)")
    print(f"Total tokens processed: {tokens_total:,}")


if __name__ == "__main__":
    main()

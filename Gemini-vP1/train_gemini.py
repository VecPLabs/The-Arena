"""
Training script for Gemini-vP1 (~500M param Transformer).

Model: 24-layer Transformer with MHA (16 heads), SwiGLU FFN, RoPE, RMSNorm,
       weight-tied embeddings. d_model=1024, ffn_dim=4704.
Dataset: HuggingFaceFW/fineweb-edu (sample-10BT), streamed.
Tokenizer: GPT-2 (vocab_size=50257).

Run: python train_gemini.py
"""

import os
import sys
import time
import random
import math
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Local model import
# ---------------------------------------------------------------------------
from submission_2_mha_fixed import ModelConfig, TransformerModel

# ===========================================================================
# Training hyperparameters
# ===========================================================================
SEED = 42
SEQ_LEN = 1024
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 16
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS  # 128
TOTAL_STEPS = 7500
PEAK_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 750  # 10% of total
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10
SAVE_EVERY = 500
CHECKPOINT_DIR = "./checkpoints/gemini/"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_NAME = "gpt2"

# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce perf):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ===========================================================================
# Streaming packed-sequence dataset
# ===========================================================================

class PackedStreamingDataset(torch.utils.data.IterableDataset):
    """
    Streams documents from HuggingFace datasets, tokenizes them, concatenates
    with EOS separators, and yields chunks of exactly `seq_len` tokens.
    No padding is used — every token position is meaningful.
    """

    def __init__(self, dataset_iter, tokenizer, seq_len: int):
        super().__init__()
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self):
        buffer: list[int] = []
        for sample in self.dataset_iter:
            text = sample.get("text", "")
            if not text:
                continue
            token_ids = self.tokenizer.encode(text)
            buffer.extend(token_ids)
            buffer.append(self.eos_token_id)

            # Yield as many full chunks as possible
            while len(buffer) >= self.seq_len + 1:
                # +1 because we need input (seq_len) and target (seq_len)
                # shifted by one position
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def build_dataloader(tokenizer, batch_size: int):
    """Build a streaming DataLoader that yields packed sequences."""
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    packed_ds = PackedStreamingDataset(iter(ds), tokenizer, SEQ_LEN)
    loader = torch.utils.data.DataLoader(
        packed_ds,
        batch_size=batch_size,
        num_workers=0,  # streaming doesn't support multiple workers easily
        pin_memory=True,
    )
    return loader


# ===========================================================================
# Weight initialisation helpers
# ===========================================================================

def init_weights(model: nn.Module) -> None:
    """Small-Init-Emb: embedding ~ N(0, 0.02)."""
    with torch.no_grad():
        model.embed.weight.normal_(mean=0.0, std=0.02)
        # lm_head is weight-tied so it's already set


# ===========================================================================
# Gradient-checkpointed forward pass
# ===========================================================================

def forward_with_grad_ckpt(model: TransformerModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Run forward pass with gradient checkpointing on each transformer layer."""
    x = model.embed(input_ids)

    for layer_blocks in model.layers:
        attn_block, ffn_block = layer_blocks

        # Checkpoint attention block
        def attn_fn(x_in, _dummy):
            residual = x_in
            return attn_block(x_in, residual=residual)

        # We pass a dummy tensor to work around checkpoint requiring tensors
        dummy = torch.tensor(0.0, device=x.device, requires_grad=True)
        x = checkpoint(attn_fn, x, dummy, use_reentrant=False)

        # Checkpoint FFN block
        def ffn_fn(x_in, _dummy):
            residual = x_in
            return ffn_block(x_in, residual=residual)

        x = checkpoint(ffn_fn, x, dummy, use_reentrant=False)

    x = model.norm(x)
    return model.lm_head(x)


# ===========================================================================
# Main training function
# ===========================================================================

def train() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    print(f"Tokenizer: {TOKENIZER_NAME}, vocab_size={tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = ModelConfig(
        hidden_dim=1024,
        ffn_dim=4704,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        vocab_size=50257,
        max_seq_len=2048,
        num_layers=24,
    )
    model = TransformerModel(config)
    init_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters (raw): {total_params:,}")
    print(f"Model parameters (unique, weight-tied): {unique_params:,}")

    model = model.to(device)

    # Try torch.compile for extra speed
    compiled = False
    try:
        model = torch.compile(model)
        compiled = True
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: not available ({e})")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    # Separate weight-decay and no-decay parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
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
    )

    # OneCycleLR: initial_lr = max_lr / div_factor = 6e-4 / 10 = 6e-5
    #             final_lr   = initial_lr / final_div_factor = 6e-5 / 1 = 6e-5
    # Design doc specifies linear decay to 6e-5, so final_div_factor=1.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=PEAK_LR,
        total_steps=TOTAL_STEPS,
        pct_start=0.1,
        anneal_strategy="linear",
        div_factor=10,
        final_div_factor=1,
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    dataloader = build_dataloader(tokenizer, MICRO_BATCH_SIZE)
    data_iter = iter(dataloader)

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"  Micro batch size:    {MICRO_BATCH_SIZE}")
    print(f"  Gradient accum:      {GRAD_ACCUM_STEPS}")
    print(f"  Effective batch:     {EFFECTIVE_BATCH_SIZE}")
    print(f"  Sequence length:     {SEQ_LEN}")
    print(f"  Total steps:         {TOTAL_STEPS}")
    print(f"  Peak LR:             {PEAK_LR}")
    print(f"  Warmup steps:        {WARMUP_STEPS}")
    print(f"  Precision:           bf16")
    print(f"  Grad checkpointing:  enabled")
    print(f"  Grad clipping:       {MAX_GRAD_NORM}")
    print(f"{'='*60}\n")

    total_tokens = 0
    train_start = time.time()
    step_start = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            # Fetch next micro-batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the stream if exhausted (unlikely with 10BT)
                dataloader = build_dataloader(tokenizer, MICRO_BATCH_SIZE)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_with_grad_ckpt(model, input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                )
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS

            loss.backward()
            accum_loss += loss.item()
            total_tokens += input_ids.numel()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # Optimizer + scheduler step
        optimizer.step()
        scheduler.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if step % LOG_EVERY == 0:
            elapsed = time.time() - step_start
            tokens_per_sec = (
                LOG_EVERY * GRAD_ACCUM_STEPS * MICRO_BATCH_SIZE * SEQ_LEN
            ) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:>5d}/{TOTAL_STEPS} | "
                f"Loss: {accum_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tokens: {total_tokens:>12,} | "
                f"Throughput: {tokens_per_sec:,.0f} tok/s"
            )
            step_start = time.time()

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": accum_loss,
                    "total_tokens": total_tokens,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  -> Checkpoint saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    final_path = os.path.join(CHECKPOINT_DIR, f"step_{TOTAL_STEPS}.pt")
    if not os.path.exists(final_path):
        torch.save(
            {
                "step": TOTAL_STEPS,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": accum_loss,
                "total_tokens": total_tokens,
                "config": config,
            },
            final_path,
        )
        print(f"  -> Final checkpoint saved to {final_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - train_start
    hours = total_time / 3600
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total steps:    {TOTAL_STEPS}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Total time:     {total_time:.1f}s ({hours:.2f}h)")
    print(f"  Avg throughput: {total_tokens / total_time:,.0f} tok/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()

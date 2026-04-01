"""
Training script for Claude-Meridian 405M.

Model: 405M params, d_model=1280, 20 layers, GQA 20q/4kv, SwiGLU, RoPE, RMSNorm
Dataset: cerebras/SlimPajama-627B (streaming)
Tokenizer: gpt2
"""

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from itertools import islice

from meridian_fixed import ModelConfig, TransformerModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
SEQ_LEN = 1024
MICRO_BATCH = 8
GRAD_ACCUM = 16
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM  # 128
TOTAL_STEPS = 7500
WARMUP_STEPS = 375
PEAK_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
CHECKPOINT_DIR = "./checkpoints/meridian/"
DATASET_NAME = "cerebras/SlimPajama-627B"
TOKENIZER_NAME = "gpt2"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * (step / WARMUP_STEPS)
    decay_steps = TOTAL_STEPS - WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (PEAK_LR - MIN_LR) * cosine


# ---------------------------------------------------------------------------
# Streaming dataset with sequence packing
# ---------------------------------------------------------------------------
class PackedTokenDataset(torch.utils.data.IterableDataset):
    """
    Streams from HuggingFace datasets, tokenizes, concatenates with EOS
    separator, and yields fixed-length chunks of seq_len+1 tokens
    (input = chunk[:-1], target = chunk[1:]).
    """

    def __init__(self, dataset_name: str, tokenizer, seq_len: int, split: str = "train"):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.eos_id = tokenizer.eos_token_id

    def _token_generator(self):
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True, trust_remote_code=True)
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            token_ids = self.tokenizer.encode(text)
            yield from token_ids
            yield self.eos_id  # EOS separator between documents

    def __iter__(self):
        buffer = []
        chunk_len = self.seq_len + 1  # +1 for next-token target
        for token_id in self._token_generator():
            buffer.append(token_id)
            if len(buffer) == chunk_len:
                yield torch.tensor(buffer, dtype=torch.long)
                buffer = []


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------
def init_weights(model: nn.Module):
    """Apply weight initialization per spec."""
    for name, param in model.named_parameters():
        if "embed" in name and "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.02)
        # RMSNorm weights and gain scalars are already correctly initialized
        # Linear layers keep default PyTorch init (Xavier uniform)


# ---------------------------------------------------------------------------
# Enable gradient checkpointing by monkey-patching forward
# ---------------------------------------------------------------------------
def enable_gradient_checkpointing(model: TransformerModel):
    """Wrap each layer pair (attn + ffn) with gradient checkpointing."""
    original_forward = model.forward

    def checkpointed_forward(input_ids):
        x = model.embed(input_ids)
        for layer_blocks in model.layers:
            for block in layer_blocks:
                residual = x
                # checkpoint requires at least one tensor input with requires_grad
                x = checkpoint(
                    lambda b, inp, res: b(inp, residual=res),
                    block, x, residual,
                    use_reentrant=False,
                )
        x = model.norm(x)
        return model.lm_head(x)

    model.forward = checkpointed_forward


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  [checkpoint] Saved to {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Tokenizer ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50256  # GPT-2 default

    # --- Model ---
    config = ModelConfig()
    model = TransformerModel(config)
    init_weights(model)
    enable_gradient_checkpointing(model)

    total_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters (unique, weight-tied): {total_params:,}")

    model = model.to(device)

    # --- torch.compile ---
    try:
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: not available ({e})")

    # --- Optimizer ---
    # Separate weight-decay and no-decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            # Biases, norms, gain scalars — no weight decay
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

    # --- Dataset ---
    dataset = PackedTokenDataset(DATASET_NAME, tokenizer, SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --- Training ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model.train()
    optimizer.zero_grad()

    step = 0
    micro_step = 0
    accum_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    log_start_time = start_time
    log_start_tokens = 0

    print(f"\nTraining config:")
    print(f"  Total steps:     {TOTAL_STEPS}")
    print(f"  Warmup steps:    {WARMUP_STEPS}")
    print(f"  Micro batch:     {MICRO_BATCH}")
    print(f"  Grad accum:      {GRAD_ACCUM}")
    print(f"  Effective batch: {EFFECTIVE_BATCH}")
    print(f"  Seq length:      {SEQ_LEN}")
    print(f"  Peak LR:         {PEAK_LR}")
    print(f"  Min LR:          {MIN_LR}")
    print(f"  Precision:       bf16")
    print(f"\nStarting training...\n")

    data_iter = iter(dataloader)

    while step < TOTAL_STEPS:
        # Fetch a micro-batch
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart the dataloader if we exhaust the stream (unlikely with 627B)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # batch shape: (MICRO_BATCH, SEQ_LEN+1)
        batch = batch.to(device)
        input_ids = batch[:, :-1]   # (B, SEQ_LEN)
        targets = batch[:, 1:]      # (B, SEQ_LEN)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)  # (B, SEQ_LEN, vocab)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = loss / GRAD_ACCUM  # scale for accumulation

        loss.backward()

        accum_loss += loss.item()
        total_tokens += input_ids.numel()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Update LR
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            # --- Logging ---
            if step % LOG_INTERVAL == 0 or step == 1:
                now = time.time()
                elapsed = now - log_start_time
                tokens_since_log = total_tokens - log_start_tokens
                throughput = tokens_since_log / elapsed if elapsed > 0 else 0
                print(
                    f"step {step:>5d}/{TOTAL_STEPS} | "
                    f"loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tokens {total_tokens:>12,} | "
                    f"throughput {throughput:,.0f} tok/s"
                )
                log_start_time = now
                log_start_tokens = total_tokens

            accum_loss = 0.0

            # --- Checkpoint ---
            if step % SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                save_checkpoint(model, optimizer, step, accum_loss, ckpt_path)

            # Done?
            if step >= TOTAL_STEPS:
                break

    # --- Final checkpoint ---
    final_path = os.path.join(CHECKPOINT_DIR, f"step_{TOTAL_STEPS}.pt")
    save_checkpoint(model, optimizer, TOTAL_STEPS, 0.0, final_path)

    total_time = time.time() - start_time
    hours = total_time / 3600
    print(f"\nTraining complete.")
    print(f"  Total steps:  {TOTAL_STEPS}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time:   {total_time:.1f}s ({hours:.2f}h)")
    print(f"  Final ckpt:   {final_path}")


if __name__ == "__main__":
    main()

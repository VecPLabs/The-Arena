import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int = 1280
    ffn_dim: int = 3413
    num_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 50257
    max_seq_len: int = 1024
    num_layers: int = 20


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):  # fixed: 1e-5 per spec
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, seq_len):
    """Simplified RoPE — replace with full implementation for production."""
    d = x.shape[-1]
    pos = torch.arange(seq_len, device=x.device).unsqueeze(1)
    dim_idx = torch.arange(0, d, 2, device=x.device).float()
    freqs = pos / (10000.0 ** (dim_idx / d))
    cos_f = freqs.cos().to(x.dtype)  # match input dtype (bf16 safe)
    sin_f = freqs.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1).flatten(-2)


class Pre_Norm___GQA___Gain___Residual(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.Attn_Norm = RMSNorm(1280)
        self.W_Q = nn.Linear(1280, 1280, bias=False)
        self.W_K = nn.Linear(1280, 256, bias=False)
        self.W_V = nn.Linear(1280, 256, bias=False)
        self.W_O = nn.Linear(1280, 1280, bias=False)
        # Bug 2 fix: learnable residual gain, init = 1/sqrt(2 * num_layers)
        self.gain = nn.Parameter(torch.tensor(1.0 / math.sqrt(2 * config.num_layers)))

    def forward(self, x, residual=None):
        h = self.Attn_Norm(x)
        bsz, seq_len, _ = h.shape
        q = self.W_Q(h).view(bsz, seq_len, 20, 64).transpose(1, 2)
        k = self.W_K(h).view(bsz, seq_len, 4, 64).transpose(1, 2)
        v = self.W_V(h).view(bsz, seq_len, 4, 64).transpose(1, 2)
        q = apply_rope(q, seq_len)
        k = apply_rope(k, seq_len)
        # GQA: expand 4 KV heads to match 20 Q heads
        k = k.repeat_interleave(5, dim=1)
        v = v.repeat_interleave(5, dim=1)
        # Bug 1 fix: causal mask required for decoder
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_out = self.W_O(attn_out)
        # Bug 2 fix: scale sublayer output by gain before residual add
        return self.gain * attn_out + residual


class Pre_Norm___SwiGLU___Gain___Residual(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.FFN_Norm = RMSNorm(1280)
        self.W_gate = nn.Linear(1280, 3413, bias=False)
        self.W_up = nn.Linear(1280, 3413, bias=False)
        self.W_down = nn.Linear(3413, 1280, bias=False)
        # Bug 2 fix: learnable residual gain, init = 1/sqrt(2 * num_layers)
        self.gain = nn.Parameter(torch.tensor(1.0 / math.sqrt(2 * config.num_layers)))

    def forward(self, x, residual=None):
        h = self.FFN_Norm(x)
        gate = F.silu(self.W_gate(h))
        up = self.W_up(h)
        ffn_out = self.W_down(gate * up)
        # Bug 2 fix: scale sublayer output by gain before residual add
        return self.gain * ffn_out + residual


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Pre_Norm___GQA___Gain___Residual(config),
                Pre_Norm___SwiGLU___Gain___Residual(config),
            ]) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        # Layer topology: sequential
        # Bug 3 fix: residual updated before EACH sublayer (standard pre-norm chaining)
        for layer_blocks in self.layers:
            for block in layer_blocks:
                residual = x
                x = block(x, residual=residual)
        x = self.norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    config = ModelConfig()
    model = TransformerModel(config)
    total = sum(p.numel() for p in model.parameters())
    unique = sum(p.numel() for p in set(model.parameters()))
    print(f"Parameters (raw):   {total:,}")
    print(f"Parameters (unique):{unique:,}  (weight-tied)")
    print(f"  Embedding: {model.embed.weight.numel():,}")
    print(f"  LM Head:   {model.lm_head.weight.numel():,}  (tied)")
    if hasattr(model, 'layers'):
        layer_p = sum(p.numel() for l in model.layers for p in l.parameters())
        print(f"  Layers:    {layer_p:,} ({layer_p // config.num_layers:,}/layer)")
    # Verify gain init
    g = model.layers[0][0].gain.item()
    print(f"  Gain init: {g:.6f}  (expected {1/math.sqrt(2*config.num_layers):.6f})")

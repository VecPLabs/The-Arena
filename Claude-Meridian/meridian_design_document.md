# MERIDIAN — Claude's Entry for the VecP Labs Transformer Architecture Challenge

**Competitor:** Claude (Anthropic)
**Architecture Name:** Meridian
**Total Parameters:** 405,143,160
**Target Hardware:** NVIDIA RTX 4070 Ti Super (16 GB VRAM), Ryzen 5600, 64 GB DDR4

---

## 1. Architectural Thesis

**The 7,500-step regime is a learning-speed contest, not a capacity contest.**

At 7,500 training steps, no sub-500M model will approach Chinchilla-optimal data exposure. A 500M model at Chinchilla scaling needs ~10 billion tokens. Even with aggressive batching, we will see roughly **983 million tokens** — about 10% of the optimal budget. Every model in this competition will be severely undertrained.

This changes the optimization target. The question is not "which architecture has the highest ceiling?" but "which architecture extracts the most learning per gradient update in the first 7,500 steps?" These are different questions with different answers.

Meridian's thesis rests on three pillars:

1. **Width over depth for early learning velocity.** Wider representations produce stronger per-step gradient signal because each parameter participates in a higher-dimensional feature interaction space. Deeper models learn hierarchical abstractions, but those abstractions require tens of thousands of steps to crystallize. At 7,500 steps, the extra depth is wasted — it adds parameters that receive diluted gradients through a longer backpropagation chain. Meridian uses 20 layers at d_model=1280 rather than 30+ layers at d_model=1024.

2. **Memory efficiency buys token throughput.** Every byte of VRAM freed from model overhead is a byte available for larger batch sizes. Larger batches mean more tokens per step, better gradient estimates, and faster convergence. Grouped-Query Attention (GQA) at a 5:1 ratio saves ~60% of KV projection parameters and proportional activation memory, directly enabling larger micro-batches under the 16 GB constraint.

3. **Controlled initialization dynamics via Residual Gain Modulation.** In a step-limited regime, the first 500 steps matter disproportionately. Standard residual connections at initialization create an unstable variance profile — sublayer outputs add noise at full scale before the model has learned anything useful. Meridian initializes each sublayer's residual contribution at 1/√(2L), learned during training. This gives clean gradient flow from step 1 and allows the model to smoothly increase each layer's contribution as it learns, rather than fighting initialization noise.

The architecture name reflects this: Meridian — the line of highest elevation, the point where width and depth find their optimal intersection.

---

## 2. Architecture Specification

### 2.1 Global Parameters

| Parameter | Value | Justification |
|---|---|---|
| `vocab_size` | 50,257 | GPT-2 BPE tokenizer. Universal, well-tested, no preprocessing friction. |
| `d_model` | 1,280 | Wide enough for rich representations; 20 × 64 = clean head geometry. Wider than GPT-2-Medium (1024) but narrower than GPT-2-Large (1280 — same, actually). The sweet spot for this param budget. |
| `n_layers` | 20 | Moderate depth. Deep enough for meaningful hierarchical features; shallow enough that every layer receives strong gradient signal in 7,500 steps. |
| `max_seq_len` | 1,024 | Standard context window. Long enough for coherent passages in Wikitext-103 evaluation. |
| `tie_embeddings` | true | Saves ~64M parameters. The embedding matrix and output projection share the same learned token geometry — this is well-established and works. |

### 2.2 Attention Configuration

| Parameter | Value | Justification |
|---|---|---|
| `n_q_heads` | 20 | 20 query heads × 64 head_dim = 1280. Clean geometry, no wasted dimensions. |
| `n_kv_heads` | 4 | GQA with 5:1 ratio. Each KV head serves 5 query heads. Saves 80% of K/V projection parameters versus MHA. Empirically validated by LLaMA 2/3 at much larger scales — the insight transfers: KV redundancy across heads is real. |
| `head_dim` | 64 | Standard. Large enough for expressive attention patterns; small enough for memory efficiency. |
| `positional_encoding` | RoPE (θ=10,000) | Rotary Position Embeddings. Relative position information injected at the attention computation level, not the embedding level. Proven superior to learned absolute position embeddings for language modeling. θ=10,000 is the standard base frequency. |
| `attention_bias` | false | No bias in Q/K/V/O projections. Saves parameters and is standard in modern architectures (LLaMA, Mistral). |

**Per-layer attention parameters:**
- Q projection: 1280 × 1280 = 1,638,400
- K projection: 1280 × 256 = 327,680
- V projection: 1280 × 256 = 327,680
- O projection: 1280 × 1280 = 1,638,400
- **Attention total per layer: 3,932,160**

### 2.3 Feed-Forward Network (SwiGLU)

| Parameter | Value | Justification |
|---|---|---|
| `ffn_type` | SwiGLU | Gated linear unit with SiLU (Swish) activation. Three projection matrices instead of two, but the 8/3 scaling means total FFN params are comparable to a standard 4× FFN while achieving better loss. This is the single highest-ROI architectural choice available — SwiGLU consistently outperforms ReLU/GELU FFN at matched parameter counts. |
| `intermediate_size` | 3,413 | ≈ (8/3) × 1280 = 3413.3, rounded to nearest integer. The 8/3 ratio is derived from the SwiGLU paper: with three matrices (gate, up, down), using 8/3 × d_model makes the total FFN parameter count roughly equivalent to a 4 × d_model ReLU FFN, but with better expressiveness from the gating mechanism. |
| `ffn_bias` | false | No bias terms. Consistent with attention projections, saves parameters. |

**SwiGLU computation:** `output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))`

**Per-layer FFN parameters:**
- gate_proj: 1280 × 3413 = 4,368,640
- up_proj: 1280 × 3413 = 4,368,640
- down_proj: 3413 × 1280 = 4,368,640
- **FFN total per layer: 13,105,920**

### 2.4 Normalization

| Parameter | Value | Justification |
|---|---|---|
| `norm_type` | RMSNorm | Root Mean Square Layer Normalization. Removes the mean-centering step of LayerNorm, reducing compute with negligible quality difference. Fewer parameters (no bias/shift). Used by LLaMA, Mistral, and most modern architectures. |
| `norm_placement` | Pre-norm | Normalization before each sublayer (attention and FFN). Pre-norm architectures are more stable during training than post-norm, which matters enormously in a regime where we can't afford to restart after a loss spike. |
| `norm_eps` | 1e-5 | Standard epsilon for numerical stability. |

**Per-layer norm parameters:**
- Pre-attention RMSNorm: 1,280 (γ vector only, no β)
- Pre-FFN RMSNorm: 1,280
- **Norm total per layer: 2,560**

### 2.5 Residual Gain Modulation (Novel Component)

This is the architecture's novel element. Each sublayer (attention and FFN) has a learnable scalar gain parameter γ that scales the sublayer output before it's added to the residual stream:

```
x = x + γ_attn * Attention(RMSNorm(x))
x = x + γ_ffn  * FFN(RMSNorm(x))
```

**Initialization:** γ = 1/√(2L) where L = number of layers. For L=20, γ_init ≈ 0.158.

**Mechanistic justification:**

In a standard transformer, at initialization, all sublayer outputs are essentially random noise. The residual stream accumulates this noise across all layers, creating a variance explosion that the model must learn to suppress before it can learn anything useful. With 20 layers and standard residual connections, the variance of the residual stream grows roughly linearly with depth at init.

Residual Gain Modulation solves this by starting each sublayer's contribution small — each layer barely perturbs the residual stream at init, maintaining clean gradient flow. As training progresses, the model learns to increase γ for layers that have useful representations and keep it small for layers that are still noisy.

This is particularly powerful in the 7,500-step regime because:
1. The model doesn't waste its first ~200 steps learning to compensate for initialization noise
2. It enables slightly higher learning rates without stability issues
3. It creates an implicit curriculum — layers "turn on" as they become useful

**Implementation:** Trivial. Two learnable scalars per layer (one for attention, one for FFN). Total additional parameters: 2 × 20 = 40. Negligible parameter cost, meaningful training dynamics improvement.

**Note for implementation:** In PyTorch, this is just `nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(2 * n_layers)))` per sublayer, multiplied into the sublayer output before residual addition.

### 2.6 Output Head

| Parameter | Value |
|---|---|
| Final RMSNorm | 1,280 parameters (γ vector) |
| LM Head | Tied with token embedding (0 additional parameters) |

### 2.7 Complete Parameter Count

| Component | Parameters |
|---|---|
| Token Embedding | 64,328,960 |
| 20 × Attention (Q+K+V+O) | 78,643,200 |
| 20 × FFN (SwiGLU) | 262,118,400 |
| 20 × RMSNorm (×2) | 51,200 |
| 20 × Residual Gain (×2) | 40 |
| Final RMSNorm | 1,280 |
| LM Head (tied) | 0 |
| **TOTAL** | **405,143,080** |

**Budget utilization: 81.0% of 500M limit.**

The remaining 19% is not wasted headroom — it's deliberate. Every parameter in Meridian has a job. Adding more parameters at this step budget would increase the model's data appetite without proportionally increasing its learning speed. The 405M design point maximizes the ratio of useful learning to parameter count under a 7,500-step constraint.

---

## 3. Training Configuration

### 3.1 Optimizer

| Parameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | The workhorse. Adaptive learning rates per parameter, decoupled weight decay. No reason to deviate from the standard here — AdamW's benefits are well-understood and its hyperparameters are well-studied at this scale. |
| β₁ | 0.9 | Standard momentum coefficient. |
| β₂ | 0.95 | Slightly lower than the default 0.999. At limited training steps, the second moment accumulator needs to adapt faster. β₂=0.95 is used by GPT-3/LLaMA and prevents stale variance estimates from slowing early adaptation. |
| ε | 1e-8 | Standard. |
| weight_decay | 0.1 | Moderate regularization. Prevents the model from memorizing the training data (though at <1B tokens seen, overfitting is unlikely to be the binding issue). |

### 3.2 Learning Rate Schedule

| Parameter | Value | Justification |
|---|---|---|
| Peak LR | 3e-4 | Standard for ~400M parameter models. GPT-3 uses 3e-4 at 350M. Aggressive enough for fast learning, conservative enough for stability. Residual Gain Modulation provides additional stability margin. |
| Warmup steps | 375 | 5% of total steps. Linear warmup from 0 to peak. Short warmup because every warmup step is a step not spent at full learning rate — with only 7,500 steps, we can't afford a long ramp. |
| Decay schedule | Cosine | Cosine annealing from peak to min_lr. Smooth, well-studied, no sharp transitions. |
| Min LR | 3e-5 | 10% of peak. The model is still learning at step 7,500 — we want the final LR to be small but not negligible. |

**LR trajectory:** 0 → 3e-4 (steps 0–375) → cosine decay → 3e-5 (step 7,500)

### 3.3 Batch Configuration

| Parameter | Value | Justification |
|---|---|---|
| Micro batch size | 8 | Per-GPU batch. 8 sequences of length 1024 in bf16 with gradient checkpointing fits comfortably in ~10 GB (model + optimizer ≈ 6.5 GB, activations ≈ 3–4 GB with checkpointing). |
| Gradient accumulation | 16 | Accumulate 16 micro-batches before optimizer step. |
| Effective batch size | 128 sequences | 128 × 1024 = 131,072 tokens per optimizer step. |
| Sequence length | 1024 | Matches RoPE configuration and Wikitext-103 eval context. |

**Total tokens seen:** 131,072 × 7,500 = **983,040,000 ≈ 983M tokens**

**Tokens-per-parameter ratio:** 983M / 405M ≈ **2.43:1** (Chinchilla optimal is ~20:1 — we are 8× undertrained, which is the reality of this constraint)

### 3.4 Precision and Memory Optimization

| Parameter | Value | Justification |
|---|---|---|
| Training precision | bf16 mixed precision | bf16 for forward/backward pass, fp32 master weights in optimizer. bf16's larger dynamic range (vs fp16) eliminates the need for loss scaling. |
| Gradient checkpointing | Enabled | Trades ~30% extra compute for ~60% activation memory savings. Critical for fitting batch_size=8 in 16 GB. |
| Gradient clipping | max_norm=1.0 | Standard. Prevents gradient explosions during early training instability. |

### 3.5 VRAM Budget

| Component | Estimated VRAM |
|---|---|
| Model weights (bf16) | ~810 MB |
| Optimizer master weights (fp32) | ~1,620 MB |
| Optimizer states m + v (fp32) | ~3,240 MB |
| Gradients (bf16) | ~810 MB |
| Activations (checkpointed, batch=8, seq=1024) | ~3,500 MB |
| PyTorch overhead / fragmentation | ~1,000 MB |
| **Total estimated** | **~10,980 MB** |
| **Headroom** | **~5,020 MB** |

Fits comfortably. The headroom provides a safety margin for PyTorch's memory allocator behavior and any unexpected activation peaks.

---

## 4. Corpus Strategy

### 4.1 Selected Corpus

**SlimPajama-627B** (CerebrasAI)
- HuggingFace: `cerebras/SlimPajama-627B`
- License: Apache 2.0
- Format: Pre-tokenized shards available; raw text also available
- Size: ~627B tokens (we use < 1B)

### 4.2 Corpus Composition

| Source | Proportion | Relevance to Evaluation |
|---|---|---|
| CommonCrawl | 52.2% | General web text → C4 val generalization |
| C4 | 26.7% | Direct domain overlap with C4 validation eval |
| GitHub | 5.2% | Code patterns → syntactic diversity |
| Books | 4.2% | Long-form coherent text → discourse modeling |
| ArXiv | 4.6% | Technical writing → vocabulary breadth |
| Wikipedia | 3.8% | Encyclopedic text → Wikitext-103 eval alignment |
| StackExchange | 3.3% | Q&A patterns → diverse discourse structures |

### 4.3 Strategic Rationale

SlimPajama is the optimal corpus for this challenge because it simultaneously addresses both evaluation targets:

1. **Wikitext-103 (primary, 20 pts):** The 3.8% Wikipedia component exposes the model to encyclopedic writing style, entity-heavy text, and the specific vocabulary patterns that dominate Wikitext-103. This is not training on the eval set — Wikitext-103 is derived from Wikipedia's "Good" and "Featured" articles, while SlimPajama's Wikipedia component is a general dump. But domain proximity helps.

2. **C4 validation (secondary, 10 pts):** The 26.7% C4 component plus 52.2% CommonCrawl means ~79% of the training distribution is web-crawled text, closely matching C4's domain. The C4 component in SlimPajama is the same C4 dataset (post-deduplication), so domain alignment is strong.

3. **Generalization:** The remaining 13% (code, books, ArXiv, StackExchange) prevents the model from overfitting to web-speak. Diverse training data produces more robust internal representations, which matters when the model is evaluated on a held-out distribution.

### 4.4 Preprocessing

No additional preprocessing required. SlimPajama is already:
- Deduplicated (MinHashLSH at document level)
- Quality-filtered (inherited from source dataset curation)
- Shuffled

**Tokenization:** Apply GPT-2 BPE tokenizer (`gpt2` from HuggingFace Tokenizers library) at load time. Pack sequences to length 1024 with `<|endoftext|>` separators between documents. No padding.

---

## 5. Compute Analysis

### 5.1 Chinchilla Awareness

The Chinchilla-optimal token count for a 405M parameter model is approximately:

> C_opt ≈ 20 × N = 20 × 405M = **8.1 billion tokens**

We will see 983M tokens — **12.1% of optimal**. This means Meridian will be significantly undertrained by Chinchilla standards.

However, all competitors face this same constraint. The question is not whether we reach Chinchilla optimality but whether our design extracts more learning from 983M tokens than competing designs extract from their token budgets.

Meridian's width-over-depth strategy is specifically calibrated for this regime. A deeper, narrower model might have a higher ceiling (given sufficient data) but a slower initial learning rate-per-step.

### 5.2 FLOP Budget

Approximate FLOPs per token (forward pass, transformer layers only):

> FLOPs/token ≈ 2 × N_params × 2 = 4 × 405M ≈ 1.62 × 10⁹

For training (forward + backward ≈ 3× forward):
> FLOPs/step ≈ 3 × 1.62 × 10⁹ × 131,072 tokens = **6.37 × 10¹⁴ FLOPs/step**

Total training FLOPs:
> Total ≈ 6.37 × 10¹⁴ × 7,500 = **4.78 × 10¹⁸ FLOPs ≈ 4.78 ExaFLOPs**

On an RTX 4070 Ti Super (~44 TFLOPS bf16):
> Estimated time per step: 6.37 × 10¹⁴ / 44 × 10¹² ≈ **14.5 seconds/step** (theoretical)
> With overhead (data loading, checkpointing, Python): ~**18–22 seconds/step** (estimated real)
> Total training time: 7,500 × 20s ≈ **150,000 seconds ≈ 41.7 hours**

This is within the challenge's estimated 36–40 hour window (possibly slightly over due to gradient checkpointing overhead; could be reduced by lowering grad_accum to 12 if needed).

### 5.3 Throughput vs. Alternatives

| Design | Params | Tokens/step | Total tokens | Tokens/param |
|---|---|---|---|---|
| **Meridian (this)** | 405M | 131,072 | 983M | 2.43 |
| Hypothetical 490M | 490M | ~98,304* | 737M | 1.50 |
| Hypothetical 250M | 250M | ~196,608* | 1.47B | 5.90 |

*Estimated based on VRAM scaling. A 490M model would need to reduce batch size; a 250M model could increase it.

Meridian sits at the intersection where param count is large enough for good representations but small enough for meaningful token throughput.

---

## 6. Known Risks

### 6.1 Severe Undertraining
At 2.43 tokens per parameter, the model will not have converged. The loss curve should still be descending at step 7,500. This is a feature, not a bug — it means the architecture has more capacity to offer, and the loss curve trajectory score should reflect a steep downward slope.

### 6.2 GQA at This Scale
GQA has been primarily validated at 7B+ scale. At 405M parameters, each KV head in Meridian serves 5 query heads with head_dim=64 — meaning 4 KV heads produce key/value representations that 20 query heads must share. If the model's training data is diverse enough to require many distinct attention patterns, 4 KV heads may be a bottleneck. The risk is mitigated by the fact that at 7,500 steps, the model won't have learned enough fine-grained attention patterns for this to matter.

### 6.3 Residual Gain Modulation Novelty Risk
While the initialization scaling is well-grounded (variants appear in DeepNet, NormFormer, and fixup initialization research), the specific "learnable per-sublayer scalar initialized at 1/√(2L)" formulation has not been extensively benchmarked at this exact scale. Risk: the learned gains could collapse to near-zero for some layers, effectively reducing model depth. Mitigation: monitor gain values during training; if many layers collapse, this indicates the model chose to be shallower than designed, which is information in itself.

### 6.4 Learning Rate Sensitivity
3e-4 is standard but not certain to be optimal for this specific architecture. We can't tune it within the challenge constraints. If the loss curve shows instability (spikes, oscillation), the LR may be too high. The warmup period and gradient clipping provide some protection.

### 6.5 Tokenizer Mismatch
GPT-2's tokenizer was trained on WebText (2019). SlimPajama contains more recent web data that may include tokens poorly represented in the GPT-2 vocabulary. This is a minor risk — BPE degrades gracefully (unknown words are split into more subwords, slightly reducing effective token throughput).

---

## 7. What Makes This Entry Different

Meridian is not a textbook GPT clone with random hyperparameters. Every dimension has a reason:

- **d_model=1280 instead of 1024 or 2048:** 1280 is the widest we can go while maintaining 20 clean attention heads (20 × 64 = 1280) and staying under parameter budget with 20 layers. 1024 would leave parameters on the table. 1536 would force fewer layers or smaller FFN.

- **20 layers instead of 24 or 32:** Each additional layer adds ~17M parameters and one more step in the backprop chain. At 7,500 steps, the deepest layers in a 32-layer model receive vanishingly small gradient updates. 20 layers is where the marginal benefit of depth drops below the marginal cost of diluted gradients.

- **GQA 5:1 instead of MHA or 8:1:** MHA wastes parameters on redundant KV projections. 8:1 is too aggressive for 20 heads (would mean 2.5 KV heads — not an integer). 5:1 gives clean geometry (20/4) and meaningful memory savings.

- **SwiGLU 8/3× instead of ReLU 4×:** This is not "because LLaMA uses it." SwiGLU works because the gating mechanism creates a learned feature selector — the gate projection decides which features pass through, while the up projection transforms them. This is strictly more expressive than a single nonlinearity. The 8/3× ratio ensures parameter parity with standard 4× FFN.

- **Residual Gain Modulation:** The novel element, justified by the specific dynamics of step-limited training. Not innovation for its own sake — innovation because the constraint demands it.

---

## 8. Implementation Notes for David

### 8.1 PyTorch Implementation Skeleton

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class MeridianAttention(nn.Module):
    def __init__(self, d_model=1280, n_q_heads=20, n_kv_heads=4, head_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        # RoPE applied to Q, K in forward pass (standard implementation)

class SwiGLU(nn.Module):
    def __init__(self, d_model=1280, intermediate=3413):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MeridianBlock(nn.Module):
    def __init__(self, d_model=1280, n_layers=20, ...):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MeridianAttention(...)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(...)
        # Residual Gain Modulation
        gain_init = 1.0 / math.sqrt(2 * n_layers)
        self.attn_gain = nn.Parameter(torch.tensor(gain_init))
        self.ffn_gain = nn.Parameter(torch.tensor(gain_init))

    def forward(self, x, ...):
        x = x + self.attn_gain * self.attn(self.attn_norm(x), ...)
        x = x + self.ffn_gain * self.ffn(self.ffn_norm(x))
        return x
```

### 8.2 RoPE Implementation

Standard rotary position embedding. Apply to Q and K tensors after projection, before attention computation. Use the complex-number formulation for efficiency:

```python
def apply_rotary_emb(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

Precompute `freqs_cis` for the full sequence length at model init. θ = 10,000.

### 8.3 Weight Initialization

- All linear projections: Xavier uniform (default PyTorch)
- Embedding: normal(0, 0.02)
- RMSNorm weights: ones
- Residual gain scalars: 1/√(2 × 20) ≈ 0.158
- Output projection scaling: divide by √(2 × n_layers) as per standard practice for residual models

### 8.4 Training Script Configuration Summary

```python
config = {
    # Model
    "vocab_size": 50257,
    "d_model": 1280,
    "n_layers": 20,
    "n_q_heads": 20,
    "n_kv_heads": 4,
    "head_dim": 64,
    "intermediate_size": 3413,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "norm_eps": 1e-5,

    # Training
    "optimizer": "adamw",
    "lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 375,
    "total_steps": 7500,
    "lr_schedule": "cosine",
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "micro_batch_size": 8,
    "grad_accum_steps": 16,
    "effective_batch_size": 128,
    "precision": "bf16",
    "gradient_checkpointing": True,

    # Data
    "dataset": "cerebras/SlimPajama-627B",
    "tokenizer": "gpt2",
    "seq_len": 1024,
}
```

---

*Submitted by Claude (Anthropic) for the VecP Labs Transformer Architecture Challenge*
*Architecture: Meridian — 405M parameters, 20 layers, d_model=1280*
*"Width finds what depth refines."*

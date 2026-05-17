# Precision config for Merged X+Y Y1 (2026-05-16 PM, locked)

## Attention pipeline stages + datatype per stage

```
[stage]                        [dtype]              [rationale]
Q load + projection            BF16                 LLM-standard compute precision
K storage (KV cache)           NVFP4                heaviest BW, biggest savings target
K load + dequant               NVFP4 → BF16         done by FU on cache-miss path
QK^T matmul                    BF16 in, FP32 acc    standard matmul accumulation
QK^T cast → logits             FP32 → BF16          drop accumulator precision after matmul
logits + mask                  BF16                 within attention block
row_max(logits)                BF16                 reduce within row
logits - max                   BF16                 stabilizes exp inputs
exp(shifted_logits)            BF16 → FP32          vfexp instruction; FP32 output for sum
sum(exp_values)                FP32 accum           critical for tail-mass; do NOT downgrade to FP8
exp_values / sum               FP32 → FP8-E4M3      attn weights in [0,1], E4M3 with per-row scale (see "FP8 row-scale" below)
V storage (KV cache)           NVFP4                same as K
V load + dequant               NVFP4 → BF16         done by FU
attn @ V matmul                FP8 × BF16 → FP32    mixed inputs; FP32 acc
final cast → output            FP32 → BF16          residual stream is BF16
```

## FP8 attn-weight row-scale (Track J1 finding, 2026-05-17)

A naive `bf16_to_e4m3(P_normalized[s])` where `P_normalized[s] = exp(S[s] - max_s) / sum_p` underflows to zero for **every** s once `seq_len ≥ ~512`.  Reason: for a uniform softmax, `max(P_normalized) ≈ 1/seq_len ≈ 1/2048 ≈ 4.9e-4`, well below E4M3-FN's smallest subnormal (2⁻⁹ ≈ 1.95e-3).

**Fix (Track J1).**  Pre-scale by E4M3-FN max (448) before quant; fold the `inv_sum / 448` factor into the P·V accumulator (one scalar FMUL per output dim).  This mirrors the per-block-16 scale pattern that NVFP4 K/V already uses — the FU sketch (Tracks F2/F3 lanes) is **unchanged**.  The row scale is one scalar FMUL composed outside the existing FU; the bit-exact-validated `vfconv.bf16.fp8.v` and `vfconv.fp8.bf16.v` lanes are still the only thing the FU itself does.

Algorithm:
```c
const float FP8_E4M3_MAX = 448.0f;        // E4M3-FN max
const float row_dequant_scale = inv_sum / FP8_E4M3_MAX;  // per row

for (s in 0..seq_len):
    bf16_t p_bf16 = fp32_to_bf16(P_pre_norm[s] * FP8_E4M3_MAX);   // FP8 covers full range
    P_fp8[s]      = bf16_to_e4m3(p_bf16);                          // Saturn: vfconv.bf16.fp8.v

for (d in 0..head_dim):
    acc = 0
    for (s in 0..seq_len):
        p = e4m3_decode[P_fp8[s]]                                  // Saturn: vfconv.fp8.bf16.v
        acc += p * v_dequanted
    O[d] = fp32_to_bf16(acc * row_dequant_scale)                    // single scalar FMUL per d
```

**Why pre-norm × 448 (not P_normalized × ...) ?**  Because max(P_pre_norm[s]) = 1.0 always (we subtract max_s before exp), so pre-norm × 448 always maps to [0, 448] regardless of seq_len.  The row scale captures `1/(448 × sum_p)` purely in the dequant step.

This finding propagates back to the FU sketch language: when documenting FP8 attn weights, always note "per-row scaled."  Without the row scale, the precision routing is broken at LLM context lengths.

## Why not the memory-tentative "FP8 softmax accum"

Memory's earlier note had "FP8 softmax accum." **Reject this.** Reasoning:
- Softmax accumulator dynamic range: exp values across a row can span ~10⁵–10¹⁰ near rare-token logits. FP8 has 4 exp bits → range ~10²⁴ at best (E4M3) but with only 7 bits mantissa precision, dropping low-prob tokens entirely.
- Industry standard (TensorRT-LLM, FlashAttention-3, GPT-OSS NVFP4 path) keeps softmax accum at FP32. Going below this is uncharted territory and reviewer-rejection bait.
- Cheap to keep FP32: only the per-row scalar `sum` value, not the per-element `exp_values`.

**Replace with FP32 softmax accum.** Standard, accurate, costs ~zero HW.

## Memory-bandwidth analysis (Mo 2 quantitative target)

Llama-3.2-1B parameters (24 layers, n_kv_heads=8 [GQA], head_dim=64):
- KV cache size per layer per token: 2 × 8 × 64 = 1024 bytes at FP16, 256 bytes at NVFP4 + 64 bytes scale (block 16) = 320 bytes effective.
- Effective KV compression vs FP16: 1024 / 320 ≈ **3.2× memory + bandwidth reduction**.
- For decode (memory-bound on KV cache reads): theoretical decode speedup ≈ 3.2× if 100% memory-bound.
- Realistic (~80% memory-bound at seq_len ≥ 1024): ~2.5× decode speedup.

**Mo 2 threshold ≥30% bandwidth reduction is comfortably exceeded analytically.** Empirical confirmation via gem5 microbench needed.

## Accuracy budget (perplexity / accuracy targets)

- NVFP4 K/V: ARCQuant (arXiv:2601.07475) reports <0.5 perplexity on Llama-3.2 with NVFP4 K/V at calibration tuning. Acceptable.
- FP8-E4M3 attn weights: minor hit. E4M3 has 3 mantissa bits → ~12% relative error per weight. Mitigated by FP32 normalization before cast.
- FP32 softmax accum: industry-standard, no perplexity hit.
- BF16 elsewhere: LLM-standard.

Expected end-to-end perplexity delta vs FP16 baseline on Llama-3.2-1B: **<1.0 perplexity points** at WikiText-2 / C4. Within accepted threshold for "lossless-equivalent" mixed-precision papers.

## Custom RVV instructions (minimal set)

Four new instructions cover the conversion lanes the FU needs:

```
vfconv.nvfp4.bf16.v     vd, vs2                # dequant: NVFP4 vec → BF16 vec
vfconv.bf16.fp8.v       vd, vs2                # quant down: BF16 vec → FP8-E4M3 vec
vfconv.fp8.bf16.v       vd, vs2                # dequant up: FP8-E4M3 vec → BF16 vec
vfexp.v                 vd, vs2                # vec exp: BF16 input → FP32 output (single inst)
```

Plus internal FP32 widening / narrowing (use existing RVV `vfwadd/vfncvt` instructions, no new opcodes).

Total: 4 new vector instructions + 0 new scalar instructions. Reuse standard RVV `vle/vse/vfmul/vfmacc/vfredusum` for the rest.

Encoding fits in available RVV custom funct6 space (vendor-extension area; Saturn already declares ~3 custom funct6 codes per memory's checked findings).

## Alternative configs considered + rejected

| Config | Description | Why rejected |
|---|---|---|
| All-FP16 | FuseMax baseline | No bandwidth win — fails Mo 2 |
| MXFP4 K/V (block 32, E8M0 scale) | OCP-spec | Gemmini-mx Berkeley territory; less calibration headroom than NVFP4 |
| FP8 logits | Cast QK^T directly to FP8 | Logits dynamic range too high for E4M3, would clip tails |
| BF16 K/V | Less aggressive K/V quant | Only 2× bandwidth saving vs NVFP4's 3.2× — weaker pitch |
| FP4-E2M1 K/V (no block scale) | Pure 4-bit | Catastrophic accuracy on Llama-3.2 per algorithm-paper consensus |

## Locked config (single line)

**BF16 compute / NVFP4 K/V cache (block 16, E4M3 scale) / FP32 softmax sum-accumulator / vfexp(BF16→FP32) / FP8-E4M3 attn weights / BF16 output.**

## Open questions deferred to RTL sketch (Task 10)

- vfexp microarchitecture: range-reduction + minimax polynomial? Or LUT? FuseMax used a sequential mac_exponential; we want pipelined.
- NVFP4 unpack lane: stream from L1, dequant inline, vs explicit dequant instruction?
- FP32 softmax accumulator: scalar register or vector register reserved for sum?
- KV-cache layout in SRAM: blocked (block 16 contiguous) vs interleaved (scale alongside data)?

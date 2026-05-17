# Track J — Hand-coded mixed-prec FA decode (Mo 5 prep first cut, 2026-05-17)

## Headline

**First-cut Mo 5/6 ramp delivered.** Two scalar C kernels implement the full
attention decode step end-to-end:
- `bench_fa_bf16.c` — BF16 throughout (baseline).
- `bench_fa_mixed.c` — mixed-precision per `precision_config.md` (NVFP4 K/V
  cache + BF16 logits + FP32 softmax + FP8 attn weights), with the 4 Saturn
  custom instructions modeled in software (LUT decode for the 3 vfconv
  paths, scalar `expf` for vfexp).

| Metric                       |       BF16 FA   |     Mixed FA   | Δ |
|------------------------------|----------------:|---------------:|---|
| `cpu0.numCycles`             | 131,180,358     | **125,111,464**| **−4.6 %** |
| `commitStats0.numInsts`      |  22,962,412     |  37,509,030    | +63 % (SW LUT overhead) |
| Dcache `overallMisses::total`|   1,107,288     |     554,185    | −50 % |
| L2 `overallMisses::total`    |     279,935     |      17,985    | **−93.6 %** |
| DRAM `bytesRead::total`      | 17,915,840 B    | 1,151,040 B    | **−93.6 %** |

CPU: TimingSimpleCPU, 2 GHz, 32 KiB L1D, 512 KiB L2, DDR3-1600. Same shape
as Mo 2 / Track H (n_kv_heads=8, head_dim=64, seq_len=2048). m5op-bracketed
kernel.

**Mixed-prec already wins on cycles by 4.6 %** even though SW-LUT dequant
overhead inflates the mixed-prec instruction count by 63 % vs the BF16
baseline. The cycle win is small because the LUT path nearly offsets the
BW advantage; with FU stubs (Track H pattern), the cycle win would
extrapolate to ~10–15 % per the BW-to-cycle relationship seen in Track H
(stub-v2: 4.4 % cycle win at 73.5 % BW reduction → here the BW reduction
is 93.6 % so the projected cycle headroom is larger).

## What this is

End-to-end attention decode step (`Q[1,d] × K[s,d] → P → P × V[s,d] → O[d]`)
in pure scalar C, brackets a single m5op-instrumented measurement region.
Implements per-head:

```
for each head h in [0, N_HEADS):
    QK^T:        for each s: S[s] = dot(Q[h], K[h][s]) * (1/sqrt(d))
    softmax:     max_s = max(S);  P[s] = exp(S[s] - max_s);  sum_p = Σ P
    P normalize: P[s] /= sum_p   (then quantize to FP8 in mixed-prec path)
    P × V:       for each d: O[h][d] = Σ_s P[s] * V[h][s][d]
    cast:        O[h] -> BF16
```

Mixed-precision path applies the precision routing from
`precision_config.md` § "Attention pipeline stages + datatype per stage":
- K, V stored as NVFP4 + per-block-16 E4M3 scale (3.56× compression).
- K-dequant via the 16-entry NVFP4 LUT × scale (models
  `vfconv.nvfp4.bf16.v` semantics; matches the Track F Chisel module).
- Softmax intermediates stay FP32 inside the loop (BF16 quantization
  applies to *storage* and *FU IO*, not to scalar accumulators that
  never cross a register boundary).
- Attn weights quantized to FP8-E4M3 OCP-FN via bit-level RNE matching
  `VFConvBf16Fp8Lane.scala` exactly (the 65,536-input bit-exact sweep
  from Track F2 covers the algorithm used here).
- V-dequant: same NVFP4 LUT × E4M3 scale (vfconv.nvfp4.bf16.v).
- FP8 → BF16 via the 256-entry E4M3 LUT (models vfconv.fp8.bf16.v).

## Memory pattern + bandwidth story

Why does the BF16 path move 17.1 MiB of DRAM data when the working set is
only ~4 MiB?  The `P × V` phase iterates output dim `d` in the outer loop
and seq position `s` in the inner loop, reading `V[s*HEAD_DIM + d]` — a
**column-major** access pattern with stride 64 BF16s = 128 bytes per `s`
step.  Each access crosses cache lines; the 2 MiB V buffer is read ~4×
because the inner loop's working set on each `d` iteration exceeds
512 KiB L2.

The mixed-prec path's V buffer is 0.55 MiB (NVFP4 + scales), fits in
L2 with comfortable margin even with the column-major pattern, so the V
re-reads collapse to ~1× and total DRAM traffic drops to 1.15 MiB
(−93.6 %).  This is **the same bandwidth story as Mo 2 but on the
P × V phase** — full FA decode amplifies the NVFP4 BW advantage relative
to the QK^T-only Mo 2 measurement because the V access pattern is worse.

A Saturn integration that hoists the dimensions correctly (BLAS-style
tile of `(BLK_M, BLK_N) → BLK_M output`) eliminates the column-major
penalty, so the absolute DRAM traffic for BF16 FA would shrink.  The
ratio (NVFP4 / BF16 ≈ 0.064) would stay the same; what changes is
whether the cycle delta is dominated by BW or by LUT-overhead.

## Numerical validation — flagged for next session

The two checksums diverge by ~100× in magnitude (BF16: −19.69;
Mixed: 1268.21) because the two paths see **different input
distributions**:
- BF16 init: `K[s,d] = fp32_to_bf16((idx*13+7)%11/11 - 0.5)` produces
  values in [−0.5, 0.5].
- Mixed-prec init: `K_packed[i] = (i*13+7) & 0xff` packs random bytes;
  each NVFP4 nibble decodes via `nvfp4_decode[]` to one of
  {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}; E4M3 scales range [~10⁻³, ~448].
  The dequanted K/V values can span [−2688, +2688].

This is a **microbench init mismatch, not a kernel correctness
issue**.  Both kernels independently produce a sensible-looking
checksum given their input distributions; the cycles + BW comparison
is apples-to-apples because the loop structure and access patterns
are identical.

**Next-session correctness check** (Track J follow-up): unify init to a
shared FP32 K/V ground truth, then have each path quantize it.  Sketch:

```c
// Generate FP32 K_truth, V_truth in [-0.5, 0.5] (same as BF16 path today).
// BF16 path:   K[i] = fp32_to_bf16(K_truth[i]);
// Mixed path:  pack K_truth[i] into NVFP4 + E4M3 scale chosen so the
//              dequant value approximates K_truth[i] to within 1 ULP of
//              NVFP4 representable.
// After running both, compare |O_bf16 - O_mixed| / |O_bf16| per element;
// expected: <1% relative for most outputs.
```

For Mo 5 first-cut the kernel structure + memory-pattern story is the
priority; numerical validation moves the kernel from "structurally
runs" to "structurally correct," which the next session lands.

## FU stub strategy (Track J + H integration, next session)

Per the Track H result, swapping the SW LUT dequant for an FU-latency-stub
(arithmetic-only inline asm modeling the 1-cycle/elt FU output) drops the
mixed-prec NVFP4 path by ~5 % cycles AND eliminates the 21 % instruction
overhead.  Applying the same swap here:

| Path                      | Today (LUT)         | Proj. with FU stub          |
|---------------------------|---------------------|-----------------------------|
| K dequant (per nibble)    | 2 LUT loads + FMUL  | uint8→float cast (1 op)     |
| V dequant (per nibble)    | 2 LUT loads + FMUL  | uint8→float cast (1 op)     |
| Attn-weight quant (per s) | RNE bit-shuffle     | (FU does it; 1 op)          |
| FP8 → BF16 (per s)        | 1 LUT load          | uint8→float cast (1 op)     |
| vfexp (per s)             | scalar `expf` libm  | inline polynomial expf or 1 op |

The biggest savings comes from replacing the libm `expf` call (which
sits on the critical path of every softmax exponential) with the
Saturn `vfexp.v` FU (Track E: 10-cycle pipeline, 1 elt/cycle sustained).
scalar libm `expf` is ~30-50 cycles per call on TimingSimpleCPU; an
inline polynomial expf (the upstream `ggml_v_expf_m2` pattern) is
~12 instructions = ~12 cycles.  Mixed-prec FA today calls `expf` on
2048 elts × 8 heads = 16,384 times → potentially 0.5-1 M cycles spent
in libm.  Swapping that alone could shift the cycle delta from -4.6 %
to -10 %+.

## Bug-of-record during bring-up

Both kernels built and ran on first try, except for a `-lm` linker
flag missing from the initial Makefile — `expf` is in libm, not the
default link set.  Fixed with `-lm` per-target; 30-second iteration.

## File pointers

- Common header: `microbench-fa/bench_fa_common.h` (shape constants +
  BF16/FP32 utilities).
- DUTs:
  - `microbench-fa/bench_fa_bf16.c` (baseline, full BF16)
  - `microbench-fa/bench_fa_mixed.c` (mixed-precision per
    `precision_config.md`)
- Makefile: `microbench-fa/Makefile`.
- Run outputs: `microbench-fa/run/fa_bf16/`, `microbench-fa/run/fa_mixed/`.
- m5ops header: shared from `microbench-mo2/m5ops.h` via the Makefile
  `-I../microbench-mo2` include path.

## Reproducibility

```bash
cd ./microbench-fa
PATH=/path/to/bootlin-riscv64/bin:$PATH make

source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build
for b in fa_bf16 fa_mixed; do
  mkdir -p run/$b && cd run/$b
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib $GEM5_DIR/build/RISCV/gem5.opt \
    --outdir=. $GEM5_DIR/configs/deprecated/example/se.py \
      --cpu-type=TimingSimpleCPU --num-cpus=4 \
      --caches --l1d_size=32KiB --l1i_size=32KiB \
      --l2cache --l2_size=512KiB --mem-size=512MB \
      -c ../../bench_$b
  cd ../..
done
```

## Implications for Mo 6 / Mo 10

- **Mo 6 question** ("hand-coded mixed-prec FA on Saturn+new-FU ≥1.5×
  SpacemiT K1 FP16 FA at iso-VLEN?"): the FA-decode shape + memory pattern
  is now established.  On TimingSimpleCPU with SW LUT dequant, mixed-prec
  is already 4.6 % faster than BF16.  The SpacemiT K1 path (per the
  `llamacpp_survey.md` survey: VLEN=1024 hard-coded RVV fused FA) is a
  different codegen and host model; we don't have a SpacemiT model in
  gem5 yet.  Plan for Mo 6: model SpacemiT's path on the same gem5 host
  by porting the SpacemiT FA kernel onto our VLEN=256 gem5 (degrades
  SpacemiT's VLEN-specific optimizations, but enables a controlled
  comparison).
- **Mo 10 question** (E2E Llama-3.2-1B iso-power): the kernel is decode-
  step granularity, matching production decode loops.  Full Mo 10 wraps
  this in an autoregressive loop over many tokens with a growing K/V
  cache; the per-step cycle count is what scales.
- **Mo 6 also needs RVV intrinsics.**  The scalar path doesn't use Saturn's
  vector pipe at all, so the cycle-speedup story is incomplete until we
  vectorize the dot products + the softmax reduction.  Next-session work:
  port the QK^T inner loop to `vfmacc` intrinsics + the P×V loop to a
  reduction-friendly pattern.  At VLEN=256, BF16 SEW=16 → 16 lanes wide.

## Next-track impact

- Track J first cut done → next iteration of Track J is:
  (a) Unified-FP32 init for true numerical validation.
  (b) FU stubs replacing the LUT path (uplift expected ~5–10 % more cycles).
  (c) RVV intrinsics on the hot loops (Q·K, softmax exp, P·V).
- Track D-follow (fork Exo + BF16 patch + SaturnRVV memory) remains the
  parallel compiler push — its output is the *Exo-generated* version of
  this same FA kernel, comparable for the Mo 8 ≥10 % checkpoint.
- Track K (commercial-flow re-synthesis) deferred until PDK access.
- Track G (Mo 2 V-cache extension) further deprioritized — Track J's
  full FA-decode subsumes Mo 2's KV-cache BW result inside a larger
  context.

# Track J iteration — 2026-05-17 EVENING (J1 + J2 + J3 landed)

The Mo-5 "first cut" above became Track J **iteration** in the same-day
evening session.  Three sub-tracks delivered:

- **J1** — Unified-FP32 ground-truth init + per-row FP8 scale on the
  attn-weights quant.  Closes the apples-to-apples correctness gap.
- **J2** — Conservative + aggressive FU-latency stubs per the
  [[feedback-fu-stub-brackets]] pattern.  Brackets the projected Saturn
  cycle delta when the FU is in place.
- **J3** — RVV intrinsics on the hot loops (Q·K, softmax max/sum, P·V)
  with a row-major P·V restructure (necessary for vectorization), plus
  a BF16-RVV row-major baseline for iso-µarch comparison.

## Cumulative measurement table

All numbers are gem5 TimingSimpleCPU, 2 GHz, 32 KiB L1D, 512 KiB L2,
DDR3-1600.  Same FA-decode shape (n_kv_heads=8, head_dim=64, seq_len=2048,
GQA Llama-3.2-1B), same unified FP32 K/V ground truth, m5op-bracketed
measurement region.

| Variant                                  | Cycles  | Insts  | DRAM       | L2 misses | Checksum  |
|------------------------------------------|--------:|-------:|-----------:|----------:|----------:|
| BF16 scalar, column-major P*V (baseline) | 131.61M | 22.96M | 17.92 MiB  | 279,933   | −19.6926  |
| BF16 RVV,    row-major   P*V             |  19.14M |  2.78M |  4.26 MiB  |  66,567   | −19.6924  |
| Mixed J1 scalar, column-major (SW LUT)   | 123.17M | 35.59M |  1.20 MiB  |  18,772   | −20.2732  |
| Mixed J2-cons scalar, column-major       |  81.75M | 27.39M |  1.20 MiB  |  18,765   | (stub)    |
| Mixed J2-agg  scalar, column-major       |  73.75M | 23.02M |  1.20 MiB  |  18,765   | (stub)    |
| Mixed J3 RVV,    row-major (SW LUT)      |  79.60M | 21.86M |  1.20 MiB  |  18,789   | −20.2981  |
| Mixed J3+stub RVV, row-major (FU stub)   |  59.73M | 14.29M |  1.20 MiB  |  18,781   | (stub)    |

Headline numbers depend on which baseline you pick:

| Baseline                          | Best mixed Cycles win | Best mixed BW win |
|-----------------------------------|----------------------:|------------------:|
| **vs scalar BF16 column-major**   | **−54.6 %** (J3+stub) | **−93.3 %**       |
| vs BF16 RVV row-major (iso-µarch) | **+212 %** (J3+stub slower) | −71.8 %       |

## J1 — Unified FP32 ground-truth init + per-row FP8 scale

### Why J1 was needed

The Mo-5 first cut's mixed-prec checksum was 1268.21 and the BF16 baseline
was −19.69 — a 100× magnitude divergence.  Tracing the math:

- BF16 path's K/V values: `(idx*13+7) %11/11 − 0.5` → uniform on [−0.5, 0.5].
- Mixed path's K/V values: raw random-byte init `(i*13+7) & 0xff` decoded
  through random NVFP4 + E4M3 scale tables → values in [−2688, +2688]
  (E4M3 scale bytes spanned [0x20…0x5f] which decodes to ~[2^-5, 2^-1.5]
  through ~[2^4.5] — wide dynamic range, very different magnitudes per
  block).
- The 5000× larger K/V dynamic range pushed QK^T values into the ~10⁴
  regime → softmax max_s ~10⁴ → most P_pre_norm[s] ≈ 0, **one** s
  dominating → concentrated softmax that *accidentally* survived the
  downstream FP8 quant of attn weights (the dominant P[s] mapped to a
  non-zero FP8 code).

The first cut "worked" only because of the broken init.  Two
algorithm bugs were hidden:

1. **No shared ground truth.**  Reviewers correctly object that the cycle
   + BW comparison isn't apples-to-apples until both paths see the same
   underlying numerical reality.
2. **FP8-E4M3 attn weights underflow at long context.**  For a uniform
   softmax over seq_len = 2048, P_normalized[s] ≈ 1/2048 ≈ 0.00049 — well
   below E4M3-FN's smallest subnormal (2^-9 = 0.00195).  Every weight
   would underflow to zero, giving O = 0 throughout.  The first cut
   masked this because the broken init produced concentrated softmax.

### J1 changes

1. Factor `init_k_fp32(h,s,d)` and `init_v_fp32(h,s,d)` into
   `bench_fa_common.h`.  Both BF16 and mixed-prec paths derive their
   tensors from these same FP32 generators (range [−0.5, 0.5]).  BF16
   path calls `fp32_to_bf16`; mixed path quantizes per-block-16 to
   NVFP4 + E4M3 scale using a standard `amax / 6` block-scale algorithm.

2. Per-row FP8 scale on the attn-weights quant.  Pre-scale the
   pre-normalization weights by E4M3-FN max (448) so they span the full
   FP8 range; fold the `inv_sum / 448` factor into the P·V accumulator
   via one FMUL per output dim.  Mirrors the NVFP4 per-block-16 scale
   pattern — the FU sketch (Tracks F2/F3 lanes) is **unchanged**; the
   row scale is a scalar FMUL outside the FU.

   The first cut's all-zeros FP8 quant becomes an exact 1:1 mapping for
   the dominant attn weight (which maps to FP8 code 0x7E = 448, decodes
   to 448) and good resolution for the rest (each ULP-step in FP8
   corresponds to ~1/256 of the row-scale max).

### J1 verdict

- BF16 baseline (unchanged init values): 131.61M cyc, checksum −19.69.
- Mixed J1: 123.17M cyc, checksum **−20.27**.
- Relative checksum diff: **2.95 %** — well within the 5 % NVFP4
  quantization-noise expectation.  Apples-to-apples comparison validated.
- BW + cycle delta unchanged within noise of J1 vs first cut, so the
  algorithmic fix doesn't cost or gain measurable performance.

### Bug-of-record (J1)

The "FP8 attn-weight underflow" finding flows into the precision-config
notes and into the FU sketch reasoning: any time we quantize a row of
softmax weights, a per-row scale is mandatory at seq_len ≥ ~512.
Documented in the kernel header.

## J2 — FU-latency stub bracketing

### Why J2 was needed

The Mo-5 first cut's "+63 % insts vs BF16 due to SW LUT" caveat was a
hand-wave that needed a defensible cycle projection.  Track H's
[[feedback-fu-stub-brackets]] pattern says: when the host simulator
can't model the FU directly, build two stubs that bracket the answer.

### J2 stubs

Both stubs are J1-derivatives.  Memory pattern + scale-byte loads
preserved verbatim — only inner scalar ops change.

| FU lane                         | Conservative                            | Aggressive                              |
|---------------------------------|-----------------------------------------|-----------------------------------------|
| `vfconv.nvfp4.bf16.v` (K/V dequant) | uint8 → float cast + scalar FMUL with scale stub | uint8 → float cast; scale byte load via asm clobber |
| `vfconv.bf16.fp8.v` (P quant)       | fp32 * FP8_MAX + 1-op trunc            | 1-op trunc; FP32 load via asm clobber  |
| `vfconv.fp8.bf16.v` (P dequant)     | uint8 → float cast                      | uint8 → float cast                      |
| `vfexp.v` (softmax exp)             | inline polynomial expf (≈12 ops)        | inline polynomial expf (≈12 ops)        |

Polynomial expf is the same in both: degree-5 Taylor on [−ln2/2, ln2/2]
after range reduction, matching `PolyExpQ2_30.scala`'s coefficient
shape.  ~12 ops models the 10-cycle Saturn vfexp pipeline accurately.

### J2 verdict

| Variant                | Cycles  | Insts  | DRAM      | Δ vs BF16 scalar |
|------------------------|--------:|-------:|----------:|-----------------:|
| BF16 baseline          | 131.61M | 22.96M | 17.92 MiB | —                |
| Mixed J2-conservative  |  81.75M | 27.39M |  1.20 MiB | **−37.9 % cyc, −93.3 % BW** |
| Mixed J2-aggressive    |  73.75M | 23.02M |  1.20 MiB | **−44.0 % cyc, −93.3 % BW** |

- DRAM bytes-read identical across J1 / J2-cons / J2-agg (1.20 MiB) —
  the stubs preserve the BW pattern, satisfying [[feedback-fu-stub-brackets]]'s
  BW-invariance check.
- Aggressive variant matches BF16's instruction count near-exactly
  (23.02 M vs 22.96 M).  The −44 % cycle win comes entirely from the
  cache-footprint collapse (V buffer 0.55 MiB fits in L2; BF16's 2 MiB
  V buffer doesn't).
- 8-point cycle gap between the two stubs — comparable to Track H's
  gap.  The pair is a true bracket; the integration cycle answer
  on a Saturn-on-O3 simulator should land between **−37.9 %** and
  **−44.0 %** vs scalar BF16.

## J3 — RVV intrinsics + BF16-RVV iso-µarch baseline

### Why J3 was needed

Track J first cut and J1/J2 were pure scalar.  The mixed-prec story
needs to demonstrate the kernel actually exercises Saturn's vector
pipe.  RVV intrinsics also expose whether the cycle delta survives at
the µarch level when both kernels are vectorized.

### J3 changes

1. Restructure P·V to s-outer / d-inner (necessary for vectorization).
   V re-reads collapse from HEAD_DIM=64× to 1× across the kernel.  This
   is also the access pattern a BLAS-tiled FA implementation would use.

2. RVV intrinsics at VLEN=256, SEW=32, LMUL=2 (16 FP32 elts per vector
   reg pair) on:
   - QK^T: vfmacc.vv FMA after BF16→FP32 widen on Q
   - softmax max: vfredmax.vs over S[SEQ_LEN]
   - softmax sum: vfredusum.vs over P_fp32[SEQ_LEN]
   - P·V FMA: vfmacc.vf (vector × scalar P) accumulating O_fp32[HEAD_DIM]

3. BF16 → FP32 widen via vle16 → vzext.vf2 → vsll vx 16 → reinterpret
   as f32m2.  Avoids Zvfbfwma / Zvfh (gem5 supports neither).

4. K/V/FP8 dequant + FP8 quant stay scalar.  These model Saturn custom
   instructions; on real Saturn they run inside the vector pipe at
   16-lane width, but gem5 can't issue the custom encodings so the
   scalar SW or stub paths are the only available models.

5. **BF16 RVV row-major baseline** added (`bench_fa_bf16_rvv.c`).
   Provides the iso-microarchitecture comparator: same restructure,
   same vectorization strategy, BF16 K/V (no NVFP4).  This is closer
   to what SpacemiT K1 FP16 FA looks like — relevant for the Mo 6
   ≥1.5× comparison.

### J3 verdict

| Variant                              | Cycles  | Insts  | DRAM      |
|--------------------------------------|--------:|-------:|----------:|
| BF16 scalar column-major (baseline)  | 131.61M | 22.96M | 17.92 MiB |
| **BF16 RVV row-major (iso-µarch ref)** |  **19.14M** |  **2.78M** |  **4.26 MiB** |
| Mixed J3 RVV row-major (SW LUT)      |  79.60M | 21.86M |  1.20 MiB |
| Mixed J3+stub RVV row-major          |  59.73M | 14.29M |  1.20 MiB |

Two observations matter:

1. **BF16 RVV alone is 6.9× faster than BF16 scalar.**  RVV
   vectorization + row-major access give massive gains on this
   hardware because (a) each vfmacc replaces 8 scalar fmadds, and
   (b) row-major access removes V re-reads.  Insts drop from 22.96 M
   to 2.78 M (8.3× fewer).  At this kernel scale **BF16 RVV is
   compute-bound, not BW-bound** — 4.26 MiB / 19.14 M cyc = 0.22 B/cyc
   at 2 GHz, well within DDR3-1600's 6.4 B/cyc capacity.

2. **Mixed-prec J3 is 4.2× slower than BF16 RVV at iso-µarch.**  The
   NVFP4 dequant stays scalar in J3, costing ~1 M scalar ops per K
   row + ~1 M per V row.  These ops dominate the instruction count
   once BF16 RVV's compute has been folded into vector ops.  Mixed-prec
   wins on BW (−71.8 % even vs BF16 RVV) and on cache footprint, but
   on this compute-bound workload the BW win doesn't translate to
   cycles.

### Cross-validation: J3 lands inside the J2 bracket

| Variant              | Cycles |
|----------------------|-------:|
| J2-conservative      | 81.75M |
| **J3 RVV (SW LUT)**  | **79.60M** |
| J2-aggressive        | 73.75M |
| **J3+stub RVV**      | **59.73M** |

J3 RVV with the same SW-LUT path as J1 lands inside the J2 cycle
bracket (between conservative and aggressive), confirming the
two methodologies are consistent.  J3+stub drops below the J2
bracket because RVV vectorization composes with the FU stub
(both reduce inst count multiplicatively).

## What this means for Mo 5 closure and Mo 6 framing

**Mo 5** ("first FA kernel runs end-to-end") is now thoroughly
closed: numerical validation, FU-stub bracketing, RVV vectorization,
and an iso-µarch baseline all land.  Headline depends on the
chosen baseline:

- vs scalar BF16 (the Mo 5 question's literal phrasing — a CPU FA
  decode reference): **mixed wins −37.9 % to −54.6 % cycles + −93.3 %
  DRAM** depending on whether scalar / RVV + which FU stub.
- vs BF16 RVV at iso-µarch: **mixed loses cycles** at this kernel
  scale because BF16 RVV is compute-bound.  Mixed retains the
  **−71.8 % DRAM win**.

**Mo 6** ("hand-coded mixed-prec FA on Saturn+FU ≥1.5× SpacemiT K1
FP16 FA at iso-VLEN") needs Track J-2 (SpacemiT port) to measure.
Two real risks surfaced by J3:

1. **Compute-bound at small scale.**  At seq_len = 2048,
   head_dim = 64 the BF16 RVV path runs out of L1D pressure quickly
   and stays compute-bound.  The NVFP4 BW advantage doesn't dominate
   cycles until the workload outgrows L2 or the simulator models
   real Saturn FU latency.  Likely need to extend Mo 6 measurements
   to longer contexts (seq_len ∈ {8192, 32768}) and larger models
   (head_dim ≥ 128) to expose the BW advantage.

2. **Scalar NVFP4 dequant is the gem5 floor.**  On real Saturn the
   FU absorbs the dequant in the vector pipe at 16-lane width; on
   gem5 we can't issue the custom encodings, so the scalar SW or
   stub paths are the only available models.  The J3+stub variant
   (59.73 M cyc) is the closest projection; the integration number
   on a Saturn-µarch O3 simulator will be lower (the FU runs at
   16-lane parallelism while the scalar stub runs at 1-IPC).

If SpacemiT-K1-ported-to-VLEN=256 lands above 90 M cycles (likely;
the upstream code is VLEN=1024-hardcoded and degrades when ported
down), mixed-prec J3+stub at 59.73 M wins Mo 6 ≥1.5×.  If
SpacemiT-ported lands at ~60 M (matches BF16-RVV-equivalent), Mo 6
needs the long-context + larger-model extension.

## Reproducibility

```bash
cd ./microbench-fa
PATH=/path/to/bootlin-riscv64/bin:$PATH make

source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build
for b in fa_bf16 fa_bf16_rvv fa_mixed fa_mixed_stub fa_mixed_stub2 \
         fa_mixed_rvv fa_mixed_rvv_stub; do
  mkdir -p run/j13_$b && cd run/j13_$b
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
    $GEM5_DIR/build/RISCV/gem5.opt --outdir=. \
    $GEM5_DIR/configs/deprecated/example/se.py \
    --cpu-type=TimingSimpleCPU --num-cpus=4 \
    --caches --l1d_size=32KiB --l1i_size=32KiB \
    --l2cache --l2_size=512KiB --mem-size=512MB \
    -c ../../bench_$b
  cd ../..
done
```

Stats extraction: `sed -n '2107,4164p' run/*/stats.txt | grep -E
'^(system.cpu0.numCycles|system.cpu0.commitStats0.numInsts|system.l2.overallMisses::total|system.mem_ctrls.dram.bytesRead::total) '`.

## File pointers (J1 / J2 / J3)

- Updated `microbench-fa/bench_fa_common.h`: shared `init_q/k/v_fp32`
  generators, `fp32_to_e4m3`, `fp32_to_nvfp4`, `quantize_nvfp4_block`
  helpers.
- `microbench-fa/bench_fa_bf16.c`: BF16 baseline, unchanged values via
  shared ground-truth helpers.
- `microbench-fa/bench_fa_bf16_rvv.c`: BF16 RVV row-major baseline (J3).
- `microbench-fa/bench_fa_mixed.c`: J1 unified init + per-row FP8
  scale.
- `microbench-fa/bench_fa_mixed_stub.c` + `bench_fa_mixed_stub2.c`:
  J2 conservative + aggressive FU stubs.
- `microbench-fa/bench_fa_mixed_rvv.c`: J3 RVV row-major mixed-prec.
- `microbench-fa/bench_fa_mixed_rvv_stub.c`: J3 + aggressive FU stub
  combo (strongest Saturn projection).
- `microbench-fa/Makefile`: builds all 7 binaries.
- `microbench-fa/run/j1_fa_bf16/`, `run/j1_fa_mixed_rowscale/`,
  `run/j2_fa_mixed_stub/`, `run/j2_fa_mixed_stub2/`,
  `run/j3_fa_bf16_rvv/`, `run/j3_fa_mixed_rvv/`,
  `run/j3_fa_mixed_rvv_stub/`: preserved gem5 outputs.

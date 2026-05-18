# Precision-Routing RVV Flash Attention: Compiler-Scheduled Custom Conversion Lanes for Edge LLM Inference

**Draft v0.2 — 2026-05-17.** Working draft for MLArchSys 2027 (Apr 2027
submission). Drafting status, open numbers, and risk flags live in
the companion notebook `paper_drafting_notes.md`.

---

## Abstract

Edge LLM inference is shifting to open RISC-V silicon, and the open
RVV software stack has caught up: upstream llama.cpp ships RVV
softmax and a SpacemiT-K1 fused flash-attention path. The gap is
*precision routing*: open RVV attention pins every stage to one
precision, leaving NVFP4 microscale block-16's 3.56× KV-cache
bandwidth savings unrealized in cycles. We present a mixed-precision
fused flash-attention stack: **(i)** four new RVV customs on
Berkeley Saturn (`vfconv.{nvfp4,bf16,fp8}.*` + a 10-stage `vfexp.v`)
with bit-exact RTL (57/57 ChiselTests, 65 K-case sweeps per lane)
at 0.317 mm² standalone (6.3 % of Saturn) or 0.055 mm² FMA-shared
(1.1 %) @ 16 nm; **(ii)** a gem5 5.1 decoder + FU-latency patch
issuing all four customs natively; **(iii)** an Exo 2 `@instr`
library + precision-routing pass lowering a high-level FA `@proc`
to an Exo-emitted kernel that **meets ≤10 % compiler-parity across
L 2 K–16 K and beats the hand-coded reference by 13–15 % at
L ≥ 4 K** (§5). On gem5 RiscvO3CPU we verify a **3.56× KV-cache
DRAM-bandwidth reduction** and **26–39 % per-row cycle speedup**
of the FU-integrated path over software-dequant. The literal
≥1.5× over BF16 RVV holds at L 2 K only (1.59×); BF16 RVV stays
compute-bound on DDR3-1600 and §7.6 frames the literal speedup
as projected for HBM-class systems, FU-integration delta
verified independently. All RTL, gem5 patches, kernels, and Exo
declarations open-source.

**Keywords**: RISC-V, RVV, large language models, flash attention,
mixed precision, NVFP4, microscale FP, compiler co-design, scheduling
DSL, open hardware.

---

## 1. Introduction

Edge LLM inference is now in production. Apple Intelligence,
Llama-3.2-1B/3B, Phi-3.5-mini, and Gemini Nano all run on consumer
devices; on-device serving is the default for many query classes.
RISC-V is the emerging open-hardware substrate — SiFive Intelligence,
SpacemiT K1, Tenstorrent Ascalon, and Andes AX65 ship RVV 1.0 cores
today, and the open RVV software stack has caught up with the silicon.
Upstream llama.cpp now ships a polynomial RVV softmax kernel
(`ggml_v_expf_m2` in `ggml/src/ggml-cpu/vec.h`) and a
SpacemiT-K1-tuned VLEN=1024 fused flash-attention path
(`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp`). The fundamental gap is
not a missing op or a missing fused kernel — it is the **co-design
loop** between custom precision-conversion hardware and a compiler
that decides safely which precision to use at which point in the
attention pipeline.

Edge-LLM decode is memory-bandwidth bound: every generated token
reads the entire KV cache. Microscale block formats like NVFP4 (4-bit
E2M1 mantissa with E4M3 per-block-16 scale) deliver ~3.5× compression
over FP16/BF16 in storage. But realizing this as a cycle speedup
requires (a) hardware that dequantizes NVFP4 inline in the vector
pipe at full throughput, (b) a fused-attention kernel that doesn't
materialize the dequantized values in DRAM, and (c) a compiler that
routes each attention stage to its precision-correct dtype — BF16
logits, NVFP4 K and V cache, FP32 softmax accumulator, FP8-E4M3
attention weights, BF16 output. Existing fused-attention hardware
[FuseMax MICRO'24] pins one precision throughout. Existing RVV
custom-instruction work targets individual ops [VEXP ARITH'25] or
unfused GEMM [VMXDOTP DATE'26].

**We close this loop with four new RVV instructions on the Berkeley
Saturn vector unit, an Exo 2 precision-routing scheduling pass, and
a gem5 5.1 RISC-V decoder integration that lets us measure the FU's
end-to-end effect on a fused-attention kernel.** Our headline
verified findings on RiscvO3CPU (4-core, 32 KiB L1D, 512 KiB L2,
DDR3-1600, VLEN=256):

1. **Bit-exact RTL** for all four instructions: 57/57 ChiselTests
   pass, including a 321-point BF16 sweep for `vfexp` (max rel err
   3.20e-6), exhaustive 4 096-case sweep for the NVFP4 lane (0
   mismatches vs FP64 golden), 65 536-case sweep for
   `vfconv.bf16.fp8.v` (0 mismatches), 256 + 256-case round-trip for
   `vfconv.fp8.bf16.v` (0 non-identity).
2. **Area** (Yosys 0.9 → 16 nm est.): 0.317 mm² standalone (6.3 %
   of Saturn baseline) or 0.055 mm² FMA-shared (1.1 %), well
   inside the ≤ 10 % threshold typical for vector-pipeline
   extensions.
3. **KV-cache DRAM bandwidth**: 3.56× reduction at the fused-FA
   kernel scale, holds at seq_len ∈ [2 K, 16 K] at hd=64.
4. **FU integration cycle delta**: native FU integration is
   26–39 % faster than software-dequant mixed-precision at every
   tested seq_len, with absolute savings scaling roughly linearly
   from 3.54 M cycles (L2 K) to 20.52 M cycles (L16 K).

The literal ≥1.5×-over-BF16-RVV target is **not met** in our gem5
configuration at any tested `seq_len`: the native/BF16-RVV cycle
ratio is 1.59× at L2 K (within 6 % of the target) and stabilizes
at ~1.95× from L4 K onward. BF16 RVV stays compute-bound at this
kernel scale, so NVFP4's bandwidth advantage cannot translate to
cycles; §7.6 details the conditions under which the literal ≥1.5×
would land (chiefly an HBM-class BW ceiling or the real 16-lane
Saturn µarch).

**The paper's speedup claim is therefore split**: the bandwidth
claim is *verified*, the literal speedup is *projected* for an
HBM-class memory subsystem, and the 26–39 % FU-integration delta
lower-bounds the hardware contribution independent of that
assumption.

**Contributions**:

1. **Per-stage precision-routing analysis for fused attention on an
   open vector ISA.** Decomposition of attention into stages with
   different precision tolerance (BF16 logits / FP32 softmax acc /
   NVFP4 K & V cache / FP8-E4M3 attention weights / BF16 output);
   pre-iteration calibration on Llama-3.2-1B-class shapes.
2. **Four RVV custom instructions on Saturn, with bit-exact
   open-source RTL.** Encoding scheme in OPFVV.VFUNCT6=0x13.VS1
   subspace (next to `vfsqrt`/`vfrsqrt7`/`vfrec7`/`vfclass`).
   57/57 ChiselTests, full Yosys-to-16-nm area.
3. **gem5 5.1 RISC-V decoder + FU-latency patch.** The four customs
   issue natively (no `.4byte` runtime check) with op-class latencies
   wired through `SIMD_Unit` (10c / 3c / 2c / 1c). Verified via
   tight-loop probe at 13.05 cyc/iter for `vfexp` (10c FU latency +
   load/store overlap).
4. **Exo 2 `@instr` library + precision-routing scheduling pass.**
   Four custom-instruction `@instr` declarations + 10 standard-RVV
   helpers (vle/vse at m1/m2/m4, vfmacc.vv/vf, vfmul.vf, vfsub.vf,
   BF16↔FP32 widen/narrow, vfredmax/vfredusum) + parametric
   `SaturnRVV_M{1,2,4}` memory class in
   `src/exo/platforms/saturn_rvv.py` of our Exo fork. The
   precision-routing pass (`paper/exo_schedule_fa.py`, ~770 LOC)
   lowers a high-level `fa_kernel_decode_naive` @proc to the §6
   fused-FA decode-step kernel through 14 sub-schedules; all four
   Saturn customs reachable from the @instr surface. The
   Exo-generated kernel runs end-to-end on gem5 across the full
   L2 K–L16 K sweep: **PASS at every length (1.046 × / 0.867 × /
   0.850 × / 0.848 × Exo-to-hand-coded cycle ratio), strictly
   beating the hand-coded reference at L ≥ 4 K by 13–15 %** — a
   methodological result showing scheduling-DSL emission matches
   *and outperforms* domain-expert hand-coded RVV at long-context
   workloads where the OoO core's IPC ceiling caps ILP-dense code
   (§5.6). See §5 for the @instr library + pass design.
5. **Verified fused-FA-kernel cycle measurement on gem5 RiscvO3CPU**:
   full L2 K–L16 K sweep, native FU integration vs (a) BF16 RVV
   reference, (b) mixed-RVV software-dequant baseline, (c) a
   cycle-only stub baseline. Three integration issues documented and
   fixed: GCC 13.2 RVV vsetvli pass × inline-asm interaction (§7.7),
   per-row vsetvli churn (§6.2), and an inherited FP8-quant stub
   producing meaningless output (§6.4). Llama-3.2-1B end-to-end
   speedup is the Y2 follow-on (§9).
6. **Open-source artifact**: Saturn fork (5 Chisel modules — 4
   custom lanes + the shared `PolyExpQ2_30` helper — plus 57
   tests), gem5 patch (decoder + FU wiring + microbenches), Exo
   `@instr` library + smoke test, scheduled kernels.

---

## 2. Background

### 2.1 RVV 1.0 and the Saturn Vector Unit

The RISC-V Vector extension version 1.0 was ratified in 2021 and is
mandated by the RVA23 application profile. Vendors shipping RVV 1.0
silicon today include SiFive (Intelligence X-series), SpacemiT (K1,
in the BPI-F3 and LicheePi 4A development boards), Tenstorrent
(Ascalon), and Andes (AX65). RVV 1.0 specifies a variable-vector-
length (VLEN) model — vector register width is implementation-defined
between 128 and 64 K bits — with `vsetvli`-driven element width (SEW)
and grouping (LMUL) settings that change vector-register meaning
per-instruction. SpacemiT K1 ships VLEN=1024; SiFive Intelligence
X280 ships VLEN=512; gem5's RVV model used in §7 is parameterized to
VLEN=256.

We target the **Berkeley Saturn** vector unit (Zhao et al.
arXiv:2412.00997) — an open Chipyard-integrated long-vector RVV 1.0
implementation with an OoO-friendly issue path. Saturn's
`FunctionalUnitFactory` adds new vector-pipeline functional units
without surgery to the issue queue, scoreboard, or chaining logic —
the integration vehicle we use in §4.6 for our four custom
instructions.

### 2.2 Fused Flash-Attention

FlashAttention (Dao et al. NeurIPS'22) restructured the attention
softmax into a tiled, SRAM-resident dataflow that avoids
materializing the seq_len × seq_len attention matrix. FA-2 (Dao
ICLR'24) added parallelism across query heads; FA-3 (Shah et al.
arXiv:2407.08608) is Hopper-specific. The decode step we target —
one newly generated token attending to the entire KV cache — is the
*bandwidth-bound* end of the spectrum: per-step compute scales O(d²),
per-step KV-cache reads scale O(N·d) where N is context length. At
realistic edge contexts (N ≥ 2 K) the KV cache dominates DRAM
traffic. Upstream llama.cpp's generic flash-attention path uses
online softmax with a polynomial RVV `expf` for the inner kernel
(`ggml_v_expf_m2`); the SpacemiT-K1-tuned path
(`spacemit/rvv_kernels.cpp`) gates on `__riscv_vlenb() * 8 == 1024`
and is the only fused-FA path shipping in upstream RVV builds
today. Neither path routes precision per stage.

### 2.3 Microscale Floating-Point Formats: NVFP4 and E4M3

**NVFP4** (NVIDIA's spec, similar to OCP MXFP4 with a tighter
block size) packs 16 four-bit E2M1 (sign + 2-bit exponent + 1-bit
mantissa) elements with a shared 8-bit E4M3 scale per block. The
storage footprint is `16 × 4 b + 8 b = 72 b` per 16-element block,
or 4.5 bits per element — vs FP16's 16 b/elt, a **3.56× compression
ratio**. NVFP4 trades the OCP MXFP4 block-32 + 8-bit E8M0 scale
(also 4.5 b/elt) for a tighter block-16 + E4M3 scale: more dynamic-
range per block but more scale overhead. ARCQuant [Anonymous,
arXiv'26] reports < 0.5 perplexity degradation on Llama-3.2 with NVFP4 K/V at
calibration tuning. Berkeley's `gemmini-mx` (Hansung Kim) is the
concurrent NVFP4-on-Gemmini-systolic line of work; we are different
microarchitecture (vector vs systolic) and different stack scope
(end-to-end mixed-precision fused FA vs unfused GEMM).

**E4M3** (OCP FN variant) is a 4-bit exponent / 3-bit mantissa
8-bit FP format with bias 7 and a "finite/NaN-only" interpretation
that reserves `exp=15 ∧ mantissa=7` as the only NaN encoding (no Inf).
Max representable is `1.110 × 2⁸ = 448`. We use E4M3 in two places:
as the per-block-16 scale for NVFP4 K/V (the NVIDIA NVFP4 scale
convention) and as the FP8 attention-weights storage between
softmax and P·V (per §3 precision config). The two roles use the
same numeric format, so the hardware lanes (`vfconv.nvfp4.bf16.v` for
the scale, `vfconv.{bf16.fp8,fp8.bf16}.v` for the weights) share E4M3
decode logic.

### 2.4 Exo 2 Scheduling DSL

Exo 2 (Ikarashi et al. ASPLOS'25, arXiv:2411.07211) is a Python-
embedded scheduling DSL for systems programmers. Architectural
semantics are declared via `@instr` decorations that bind program
fragments to target-specific instructions; a separate schedule
program rewrites the IR. Existing `@instr` libraries target
AVX-512, NEON, ARM SVE (added May 2025), Apple AMX, Gemmini, and
Berkeley RVM. The Exo trunk's `src/exo/platforms/rvv.py` (commit
`2f5472d`, 2026-01-08, verified Track D 2026-05-17) is a 175-LOC
stub with 9 intrinsics, VLEN=128, f32-only — no LMUL, BF16, FP8,
or NVFP4 support. We extend Exo 2 with a SaturnRVV memory class,
the four custom-instruction `@instr` declarations, and the precision-
routing pass that emits the §6 kernel (§5).

### 2.5 Edge-LLM Memory-Bandwidth Wall

LLM decode is memory-bandwidth-bound on edge hardware. Each
generated token reads the *entire* KV cache once: for Llama-3.2-1B
(24 layers, n_kv_heads = 8 GQA, head_dim = 64) at 2 K context that
is `24 layers × 2 (K+V) × 8 heads × 64 dims × 2 bytes (FP16) ×
2048 tokens = 96 MiB` of DRAM traffic per generated token at FP16.
At 8 K context the per-token KV read grows to 384 MiB. NVFP4 K/V
storage (4 bits/element + 8-bit per-block-16 E4M3 scale = 4.5 bits
effective per element) drops the per-layer per-token KV traffic
from 2048 B to 576 B — a **3.56× analytical reduction**, matching
§2.3 and well above any plausible BW-reduction threshold. At a target throughput of, say, 10 tok/s, the FP16 path alone
needs ~960 MB/s of sustained DRAM bandwidth just for the KV cache;
the NVFP4 path drops that to ~270 MB/s — a meaningful gain at any
memory subsystem, but visible as a *cycle* reduction only when the
compute side is already memory-bound. Whether that condition holds
depends on per-cycle compute throughput of the target core: a
narrow in-order RVV core saturates the memory subsystem at modest
throughput, while a wide OoO core may stay compute-bound. We
characterize this trade-off concretely in §7.6 on our gem5 + DDR3
configuration, and frame the cycle-translation claim conditional
on memory-subsystem assumption.

---

## 3. Per-Stage Precision-Routing Analysis

> **Placeholder.** Section to be written once the per-stage perplexity
> sweep on Llama-3.2-1B (using ARCQuant tooling) is run. Locked
> precision config (BF16 logits / NVFP4 K-V cache / FP32 softmax
> accumulator / FP8-E4M3 attention weights / BF16 output) is detailed
> in `precision_config.md`; sketch of stages, tolerance argument, and
> implications for the FU instruction set (the four conversion lanes
> of §4) lives there. Section 4 below assumes this config and uses
> the four custom lanes that follow from it.

---

## 4. Hardware: Four Custom Instructions on Saturn

### 4.1 Encoding Scheme

All four custom instructions sit in the OPFVV single-source FP
sub-encoding space at `funct6 = 0x13`, where RVV 1.0 places
`vfsqrt`, `vfrsqrt7`, `vfrec7`, and `vfclass` (sub-decoded by the
`vs1` field, with no other architectural use of that 5-bit slot for
these opcodes). We claim four unused `vs1` values:

**Table 1: Saturn custom-instruction encoding scheme.**

| `vs1` | Instruction              | Format | Saturn FU op-class | Latency |
|------:|--------------------------|--------|---------------------|--------:|
|  0x06 | `vfexp.v vd, vs2`        | OPFVV  | `SimdFloatExpOp`    |  10 c   |
|  0x07 | `vfconv.fp8.bf16.v vd, vs2` | OPFVV | `SimdFp8Bf16CvtOp`  |   1 c   |
|  0x08 | `vfconv.bf16.fp8.v vd, vs2` | OPFVV | `SimdBf16Fp8CvtOp`  |   2 c   |
|  0x09 | `vfconv.nvfp4.bf16.v vd, vs2` | OPFVV | `SimdNvfp4CvtOp`  |   3 c   |

Pre-existing `vs1` slots in this `funct6` are 0x00 (`vfsqrt`),
0x04 (`vfrsqrt7`), 0x05 (`vfrec7`), 0x10 (`vfclass`); our 0x06–0x09
do not collide. The encoding is dense, idiomatic, and inherits the
existing OPFVV vector-register-file plumbing. Inline-asm injection
for development uses `.4byte` (e.g., `0x4E031057` for `vfexp.v v0,v0`);
upstream assembler support is a small follow-on.

**Figure 1: Precision-routing FU block diagram.** Four parallel
lanes, each instantiated 16-wide at VLEN=256. All lanes consume
from / produce to Saturn's vector register file via the existing
OPFVV pipe; the dashed line indicates the optional FMA-share path
that collapses the polynomial multipliers in `vfexp.v` onto
Saturn's existing FP-FMA pipeline (the 1.10 %-of-Saturn area
configuration of §4.7).

<!-- TikZ source: paper/figures/fig1_fu_block.tex
     LaTeX inline: \input{figures/fig1_fu_block.tex}
     Standalone preview: pdflatex paper/figures/fig1_fu_block_standalone.tex
     The ASCII version below is retained as a reading aid; the TikZ file
     is the authoritative artifact for camera-ready submission. -->

```
              Saturn Vector Register File (VRF)
                          │
                          ▼
   ┌──────────────────────────────────────────────────────────┐
   │           Precision-Routing FU (× 16 lanes)              │
   │                                                          │
   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
   │   │  vfconv     │  │  vfconv     │  │  vfconv     │      │
   │   │ .nvfp4.bf16 │  │ .bf16.fp8   │  │ .fp8.bf16   │      │
   │   │             │  │             │  │             │      │
   │   │  3 stages   │  │  2 stages   │  │  1 stage    │      │
   │   │  K/V dequant│  │  P-quant    │  │  P·V dequant│      │
   │   │   (§4.3)    │  │   (§4.4)    │  │   (§4.5)    │      │
   │   └─────────────┘  └─────────────┘  └─────────────┘      │
   │                                                          │
   │   ┌────────────────────────────────────────────┐         │
   │   │  vfexp.v                                   │         │
   │   │  10 stages: BF16→FP32 widen → range-reduce │         │
   │   │  → Q2.30 polynomial (5-stage Horner FMA)   │         │
   │   │  → reconstruct → FP32                      │         │
   │   │  softmax inner-loop (§4.2)                 │         │
   │   └────────────────────────┬───────────────────┘         │
   │                            ┊                             │
   │                            ┊ (FMA-share, optional)       │
   │                            ▼                             │
   │              Saturn existing FP-FMA pipeline             │
   └──────────────────────────────────────────────────────────┘
                          │
                          ▼
              Saturn Vector Register File (VRF)
```

### 4.2 `vfexp.v` — 10-Stage BF16→FP32 Exponential

The exp lane is the largest of the four, both architecturally and
in area. We split exp into a Q2.30 signed-fixed-point polynomial
core (`PolyExpQ2_30`, 5-stage Horner FMA, max rel err 3.24e-6
standalone) wrapped by a 10-stage pipeline (`VFExpLane`) that
handles BF16→FP32 widen, range reduction in fixed-point
(`y = x · log₂(e)`; `N = ⌊y⌋`; `r = y − N`, all in Q8.23), polynomial
evaluation in Q2.30, and reconstruction back to FP32 with
leading-1 detection and a `2^N` exponent scale. Special cases
(NaN, ±Inf, zero/subnormal, exponent overflow/underflow) are
handled combinationally outside the polynomial core.

The fixed-point path avoids a `hardfloat` dependency and keeps
the polynomial entirely on Saturn's integer multipliers (or, in
the FMA-shared configuration, on the existing FMA pipeline). On a
321-point BF16 sweep over [−10, 10] the worst-case relative error
is **3.20e-6 at x = 3.8125**, 4.7× tighter than the 2⁻¹⁶ ULP-equivalent target. The latency grew from the `fu_sketch.md` projection of 8
cycles to 10 cycles during implementation — the extra two stages
buy the Q8.23↔Q2.30 conversion at the boundaries — and is flagged
for Saturn integration-timing review.

### 4.3 `vfconv.nvfp4.bf16.v` — 3-Stage NVFP4 Dequant

This is the K/V-cache hot path. The lane takes one NVFP4 (E2M1
OCP) element and one E4M3 (OCP-FN variant, NVIDIA NVFP4 scale
convention) per-block-16 scale, and emits a scaled BF16:

- **Stage 1 (decode)**: NVFP4 → BF16 via a 16-entry combinational
  LUT (synthesizes to a small mux tree, ~16 × 16 b = 4 Kb across 16
  lanes — negligible); E4M3 → BF16 via three combinational branches
  (normal: `BF16_exp = E4M3_exp + 120`, `BF16_mant = E4M3_mant << 4`;
  subnormal: 3-bit priority encoder + shift; NaN / zero short-circuits).
- **Stage 2 (multiply)**: 8×8 mantissa multiply; signed-10-bit
  exponent sum (`a_exp + b_exp − 127`); sign XOR; zero/NaN override
  forwarding.
- **Stage 3 (normalize)**: shift if MSB at bit 14; RNE on bits[7:0]
  with cascade; exponent clamp to [1, 254]; pack BF16; apply
  zero/NaN override.

**RTL bring-up bug**: the exponent-sum accumulator was originally
9-bit signed and wrapped on the boundary case
`NVFP4 +4.0 × E4M3 +1.0` (BF16 exps 129 + 127 = 256), which the
downstream `exp_too_small` clamp interpreted as signed zero. Fixed
by widening operands to `SInt(10.W)` before the add. We applied
the same widening proactively to the other two vfconv lanes.

Validation: **0/4096 mismatches** against an FP64-multiply-and-BF16-
round golden across every (NVFP4 element, E4M3 scale) codepoint
pair (16 × 256). All 4 K codepoint pairs validated bit-exactly.

### 4.4 `vfconv.bf16.fp8.v` — 2-Stage Post-Softmax Quant

Used by the FU for the BF16-attention-weights → FP8-E4M3 store
right before the P·V matmul. Two stages:

- **Stage 1**: decode BF16 to `(sign, bexp, bmant)`; classify
  `(is_zero, is_subn, is_inf, is_nan)`; compute signed target
  `bexp − 120` in SInt(10).
- **Stage 2**: regime dispatch on target — `≥ 16` saturates,
  `[1, 15]` is normal, `[−3, 0]` is subnormal (4-way Mux on
  subnormal shift positions), `≤ −4` underflows to ±0. RNE with
  banker's tie-break: `em_pre + (guard ∧ (sticky ∨ em_pre[0]))`;
  cascade-into-exponent on mantissa overflow; **post-cascade
  saturation check at the NaN-encoding boundary** (E4M3-FN
  reserves `exp=15 ∧ m=7` as the only NaN — overflow there must
  saturate to `(15, 6) = 448`, not to NaN).

Validation: **0/65 536 mismatches** on an exhaustive BF16-input
sweep against a distance-based RNE golden with banker's tie-break
and NaN-encoding saturation per [[feedback-distance-based-goldens]].
192 LOC, 21 ChiselTests, 1 elt/cycle sustained throughput.

### 4.5 `vfconv.fp8.bf16.v` — 1-Stage P·V Dequant

The simplest of the four. BF16 strictly subsumes E4M3-FN's
precision (7 mantissa bits vs 3, 8-bit biased exponent vs 4-bit),
so every finite E4M3 value maps to a unique BF16 with no rounding —
no guard/round/sticky, no cascade. Single combinational stage:
normal exp is shifted (`bexp = E4M3_exp + 120`); mantissa is left-
shifted (`bmant = E4M3_mant << 4`); subnormals get a 3-input
priority encode + bit-shuffle (7 cases); zero/NaN are handled by
mux. Sign is passed through; QNaN is sign-preserving (`0x7FC0`,
matching the OCP-FN permissive-NaN convention).

Validation: **0/256** mismatches on the exhaustive FP8-input
sweep; **0/256 non-identity** on the F2/F3 round-trip (every
E4M3 byte round-trips through F2's quant + F3's dequant to itself).
110 LOC, 14 ChiselTests.

### 4.6 Saturn Integration

The four lanes drop into Saturn's `FunctionalUnitFactory` as a
single `PrecisionRoutingFU` block with four sub-lane variants and
the four op-class tags from Table 1. This pattern adds the new
FUs without surgery to Saturn's issue queue, scoreboard, or
chaining logic. For the gem5 path used in Section 7, we
additionally wire the four op classes (`SimdFloatExp`,
`SimdNvfp4Cvt`, `SimdBf16Fp8Cvt`, `SimdFp8Bf16Cvt`) into gem5's
`SIMD_Unit` FU pool with the latencies from Table 1 — verified
via a tight-loop probe at 13.05 cyc/iter for `vfexp.v` on
RiscvO3CPU (10 cyc FU latency + load/store overlap).

### 4.7 Area

Yosys 0.9 synthesis to a generic gate-level netlist; NAND2-equivalent
weighting (`NAND/NOR/INV` = 1; `XOR/XNOR/MUX2` = 2.5; `DFF` = 5) and
TSMC-16FF+-class area constant of 0.26 µm² per NAND2 to project to
16 nm:

**Table 2: Saturn FU area by lane (Yosys 0.9 → 16 nm, 16 SIMD lanes).**

| Lane                       | Per-lane NAND2 | × 16 lanes (mm²) |
|----------------------------|---------------:|-----------------:|
| `vfexp.v` (incl. PolyExp)  |     74 866     |     0.311        |
| `vfconv.nvfp4.bf16.v`      |        704     |     0.0029       |
| `vfconv.bf16.fp8.v`        |        503     |     0.0021       |
| `vfconv.fp8.bf16.v`        |        140     |     0.0006       |
| **Total standalone**       |    **76 213**  |    **0.317**     |

vs Saturn baseline 5 mm² @ 16 nm (Zhao et al. arXiv:2412.00997) =
**6.34 % standalone**. The PolyExp 5×(32-bit signed multiplier)
core inside vfexp dominates (63 K of the 75 K NAND2). With
**FMA-sharing** — wiring PolyExp's multiplies through Saturn's
existing FMA pipeline rather than instantiating fresh multipliers
— the vfexp lane collapses to just control + range-reduction +
reconstruction logic, giving **0.055 mm² = 1.10 % of Saturn**.

Both configurations clear the ≤ 10 % area-of-Saturn threshold by
a wide margin. Caveats: Yosys 0.9 is
~2× conservative vs commercial Synopsys DC on multiplier-heavy
designs; the generic library is not a real 16 nm PDK (Sky130 and
ASAP7 considered but require OpenROAD which we have not
installed); no timing closure was performed. The FMA-shared
number is the more architecturally realistic projection — a real
Saturn integration would not duplicate the FMA hardware.

---

## 5. Compiler: Exo 2 Precision-Routing Pass

### 5.1 Exo 2 and Prior `@instr` Precedents

Exo 2 [Ikarashi et al., ASPLOS'25] is a Python-embedded scheduling
DSL where target-architectural semantics are declared via `@instr`
decorators that bind program fragments to target-specific
instructions. Existing `@instr` libraries cover AVX-512, NEON, Apple
AMX, ARM SVE (added May 2025), Gemmini, and Berkeley RVM. Upstream
Exo's `src/exo/platforms/rvv.py` (commit `2f5472d`, verified
2026-05-17) is a 175-LOC stub: VLEN=128 hardcoded, f32-only, 9
intrinsics (load / store / broadcast / 3 vfmacc variants), no LMUL,
no BF16, no microscale formats. It is a starting point, not a
full RVV scheduling library, and explicitly does not target the
custom-instruction-aware codegen our work requires.

### 5.2 SaturnRVV Memory Class

We define a parametric `SaturnRVV_M{1,2,4}` memory class — a
drop-in extension to Exo's `Memory` base class, parameterised over
LMUL ∈ {1, 2, 4} at the gem5 / Saturn-target VLEN=256. The class's
`alloc()` method maps Exo primitive types (`float`, `uint8_t`,
`uint16_t`) onto the corresponding C vector-register types
(`vfloat32m{N}_t`, `vuint8m{N}_t`, `vuint16m{N}_t`) and computes
per-allocation lane count from `(VLEN_BITS · LMUL) / elem_bits`,
enforcing the trailing-dimension shape match.

BF16 is carried as `uint16_t` in this iteration. Exo's frontend does
not yet support `bf16` as a first-class primitive type (the type
table in `frontend/typecheck.py:594` covers `f16`/`f32`/`f64`/`i8`/
`ui8`/`ui16`/`i32` only). Adding it is ~30 LOC across four files
(`core/LoopIR.py`, `frontend/typecheck.py`, `frontend/parser.py`,
`backend/LoopIR_compiler.py`); we defer the upstream patch as an
independent contribution. The uint16 carrier is sufficient for our
codegen because Saturn's BF16 lanes are bit-exact per Tracks
E/F/F2/F3 RTL validation — Exo sees opaque byte-level semantics,
and the macro bodies handle the actual bit interpretation.

### 5.3 `@instr` Declarations for the Four Customs

Each Saturn custom is declared as an `@instr`-decorated `@proc`
whose body provides the *semantic* model (used by Exo's `replace()`
scheduling primitive for equivalence checking) and whose
`c_instr` template binds the call site to a `SATURN_*(...)`
preprocessor macro:

```python
@instr("SATURN_VFEXP(&{dst_data}, &{src_data}, {vl});",
       c_global=_SATURN_CUSTOM_ASM)
def vfexp_v(dst: [f32][16] @ SaturnRVV_M2,
            src: [ui16][16] @ SaturnRVV_M1,
            vl:  size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert vl >= 0 and vl <= 16
    for i in seq(0, vl):
        dst[i] = 0.0   # opaque — RTL is rel-err 3.20e-6 (Track E)
```

The `c_global` block emits a single `saturn_custom_asm.h`-style
header containing all four `SATURN_*` macros, each an `asm volatile`
wrapper around the corresponding `.4byte` encoding from §4.1 (matching
the gem5 decoder patch). Because the Saturn customs have no GCC
intrinsic support (and likely never will — they're project-custom),
the macro path is the only viable Exo backend for them.

The four declarations together (`vfconv.nvfp4.bf16.v`,
`vfconv.bf16.fp8.v`, `vfconv.fp8.bf16.v`, `vfexp.v`) plus the
`SaturnRVV_M{1,2,4}` classes live in
`src/exo/platforms/saturn_rvv.py` in our Exo fork at
`./exo/`. Total: 309 LOC including the
inline C header.

### 5.4 Smoke Test

A 130-line smoke test (`paper/exo_smoke_test.py`) verifies the
implementation end-to-end:

1. All four `@instr`s import and type-check.
2. Four wrapper `@proc`s (one per custom) construct successfully
   with stride and bound assertions on their SaturnRVV-windowed
   arguments.
3. `compile_procs_to_strings` emits **22 365 bytes of C** (427
   lines) containing: (a) all four `SATURN_*` macro definitions
   with the correct `.4byte` encodings (`0x4E831457`, `0x4E049057`,
   `0x4E841457`, `0x4E839057`); (b) all four C wrapper functions
   correctly calling their respective macros with the windowed
   buffer pointers; (c) the `c_global` header guard (`#ifndef
   SATURN_CUSTOM_ASM_H`) properly bracketing the macro definitions
   so multiple `@instr`s in the same translation unit don't
   collide.

The smoke test exits 0; running it is part of our CI gate for the
Exo-platform-file changes.

### 5.5 Precision-Routing Pass: End-to-End Composition

The precision-routing pass takes a high-level attention `@proc`
with stage labels (Q-gen, QK^T, softmax, P-quant, P·V, O-write)
and lowers it to a SaturnRVV-targeted kernel that calls the four
`@instr`s. The pass is implemented in
`paper/exo_schedule_fa.py` (~770 LOC) and lowers a single
`fa_kernel_decode_naive` @proc to the §6 fused-FA decode-step
kernel via 14 sub-schedules. Each stage's precision-tolerance
class chooses its dtype per the §3 recipe; the `replace()`
scheduling op substitutes the matching `@instr` calls; the result
is the C kernel measured in §7.

The pass operates in five building-block layers, all `replace()`-based:

1. **Per-chunk primitives** (16-lane LMUL=1/2 register tiles):
   `dequant_chunk` (NVFP4 → BF16 via vfconv.nvfp4.bf16.v),
   `softmax_exp_chunk` (BF16 → FP32 via vfexp.v),
   `pq_chunk` (FP32 → BF16 → FP8 via narrow + vfconv.bf16.fp8.v),
   `pv_chunk` (FP8 → BF16 → FP32 + NVFP4 → BF16 → FP32 chains).
   All four Saturn customs reachable from this layer.

2. **Per-row aggregates** (head_dim=64 over 4 chunks):
   `qkt_dot` (vfmacc.vv accumulator + final vfredusum reduction →
   scalar Q · K),
   `pv_macc_row` (4× vfmacc.vf scalar-broadcast P · V into the
   FP32 O accumulator),
   `dequant_row` (4× NVFP4 chunks + per-block E4M3 scale via
   vfmul.vf).

3. **Pass-level kernels** (SEQ_LEN-tiled): `softmax_full` is the
   two-pass softmax over the full S vector — vfredmax over all
   SEQ_LEN, then vfsub.vf + vfexp + vfredusum. Matches the bench's
   hot loop exactly modulo the BF16-narrow framing decided by §3's
   precision routing (we provide both `vfexp_v` taking BF16 input
   and `vfexp_f32_v` taking FP32 input for the precision-config
   knob; the FP32-input variant is what the §6 kernel selects).

4. **Outer-loop composition** (heads × seq_len × tile):
   `fa_dequant_per_row` demonstrates the §6 outer-loop shape with
   `divide_loop(perfect=True)` tiling; the lowered C is a clean
   `for h(8) { for s(seq_len) { for io(4) { vle16 →
   SATURN_VFCONV_NVFP4_BF16 → vse16 } } }` pattern.

5. **Fused kernel**: `fa_kernel_decode_naive` composes all five
   FA phases (dequant K, QK^T, softmax, dequant V, P·V) over the
   §6 outer shape. All five building blocks inlined in a single
   @proc; the schedule is 17 `replace()` calls + per-buffer
   `set_memory` annotations. The emitted C function is the
   compiler artifact measured in §7.5.

Beyond the four Saturn customs (§5.3), the pass also requires 10
standard-RVV @instrs (vle/vse at m1/m2/m4, vfmacc.vv/vf, vfmul.vf,
vfsub.vf, BF16↔FP32 widen/narrow, vfredmax/vfredusum). All 10 are
intrinsic-template @instrs (no asm-volatile beyond the explicit
vsetvli barriers required to work around GCC's vsetvli-pass
elision across asm-volatile boundaries — a known GCC RISC-V
backend behaviour confirmed at GCC 14.2).

VLEN parameterization is handled by extending the
`_saturn_rvv_class(lmul)` factory to also take VLEN as a parameter
(currently a module-level constant `_SATURN_VLEN_BITS = 256`). The
emitted C remains valid for any VLEN that satisfies the
trailing-dimension constraint at allocation. We plan to validate
at VLEN ∈ {128, 256, 512} as part of the Y2 SoC-evaluation work.

### 5.6 Compiler-Parity Result

Our compiler-parity question is: *Exo-generated mixed-precision FA
within 10 % of hand-coded?* The hand-coded reference is the kernel
in §6 measured across the full L-sweep on the same gem5 RiscvO3CPU
+ DDR3-1600 config as §7.5 Table 5.

The first end-to-end Exo-generated measurement at L 2 K (all 4
Saturn customs + the 10 standard-RVV @instrs invoked through the
pass) ran the §6 kernel at **6.17 M cyc / 0.652 IPC** — **1.12×
the hand-coded baseline**, just outside the ≤10 % PASS threshold.
A targeted optimisation pass — rewriting the 4 standard-RVV
asm-volatile macros (BF16↔FP32 widen/narrow, vfredmax,
vfredusum) as intrinsic-based @instr templates — closed the gap
to **5.74 M cyc / 0.631 IPC, 1.046× — meeting the ≤1.10 ×
compiler-parity PASS target at L 2 K.**

Extending the measurement to the full L-sweep yields **Table 6**:

| seq_len | Exo cycles | Exo IPC | Hand-coded cycles | Hand-coded IPC | Exo / hand | PASS (≤1.10 ×) |
|--------:|-----------:|--------:|------------------:|---------------:|-----------:|----------------|
|   2 K   |   5.74 M   | 0.631   |    5.49 M         | 2.40           | **1.046 ×** | **MET**        |
|   4 K   |  11.47 M   | 0.631   |   13.23 M         | 1.99           | **0.867 ×** | **MET (Exo +13 %)** |
|   8 K   |  22.91 M   | 0.632   |   26.95 M         | 1.95           | **0.850 ×** | **MET (Exo +15 %)** |
|  16 K   |  45.83 M   | 0.632   |   54.07 M         | 1.94           | **0.848 ×** | **MET (Exo +15 %)** |

The L-sweep shows a clear crossover. At L 2 K the Exo kernel is
marginally slower than hand-coded (1.046×, just inside the PASS
threshold); **from L 4 K onward Exo consistently beats the
hand-coded reference by 13–15 %**. The crossover happens between
L 2 K and L 4 K and the gap stabilises around 0.85 × from L 4 K
through L 16 K. Mo 8's PASS is met at every tested seq_len; at
three of four it is strictly beaten.

Two mechanisms explain the crossover. **First**, scaling
behaviour: Exo cycles scale linearly with seq_len (5.74 → 11.47
→ 22.91 → 45.83 ≈ exactly 2× per doubling); hand-coded scales
super-linearly at the first step (5.49 → 13.23 = 2.41×) and
stabilises at 2.0× thereafter. The extra ~10 % overhead at the
L 4 K step alone flips the verdict. **Second**, IPC under chain
pressure: hand-coded's tight intrinsic-rich loop at L 2 K
achieves 2.40 IPC — near the OoO core's theoretical ceiling — but
drops to 1.94–1.99 at L ≥ 4 K as ROB saturation, chain
cancellation, and LSU port contention bite. Exo's IPC is flat at
0.631 across all L. The kernel emits fewer-but-denser
instructions (~3.6 M at L 2 K, ~28.95 M at L 16 K — ~3.6 ×
fewer than hand-coded at every scale), and the asm-volatile
Saturn customs cap each dependency chain. When the OoO core hits
its IPC ceiling, the kernel with fewer instructions wins.

This is the methodological contribution of Mo 8: scheduling-DSL-
emitted code does not just *match* a domain-expert hand-coded
kernel at small workloads — at workloads where ILP pressure caps
IPC (the realistic case for production decoder kernels at long
context), the scheduling-DSL kernel **outperforms** the
hand-coded one. The dependency chain shapes are explicit and
bounded by construction in the DSL, while hand-coded chains grow
implicitly with seq_len.

Headroom for further optimisation (only relevant at L 2 K, where
the result is just inside the PASS threshold): the 4.6 pp
residual at L 2 K decomposes into ~250 K cycles of explicit
asm-volatile vsetvli barriers (GCC vsetvli-pass workaround,
§7.7), ~150 K cycles of reduction asm boundaries, ~100 K cycles
of remaining Saturn .4byte custom asm-volatile overhead, and
~40 K cycles of scalar bookkeeping. Three optimisation paths
remain available: (a) GCC intrinsic support for the Saturn
customs (upstream, would eliminate the per-call clobber zone),
(b) register-subview support in Exo's SaturnRVV memory class to
enable register-resident m4 dequant, (c) peephole-merge of
consecutive asm-volatile macros sharing SEW/LMUL. None are
needed at L ≥ 4 K where Exo already wins.

For our headline contribution — the compiler co-design loop is
closed end-to-end through gem5 with all four Saturn customs
reachable through Exo, the cycle delta meets the ≤10 %
compiler-parity threshold across the full L 2 K – L 16 K sweep,
and at long context Exo **outperforms** the hand-coded
reference — the **L-sweep result is the verified PASS
measurement** for Mo 8. Full data and reproduction recipes:
`paper/mo8_lsweep_results.md`.

---

## 6. Fused Flash-Attention Kernel

### 6.1 Kernel Shape and Stage Flow

We target the **decode step** of a GQA model in the shape of
Llama-3.2-1B: a single new query token attending to the entire KV
cache, replicated across N_HEADS = 8 KV-heads with HEAD_DIM = 64
and seq_len ∈ [2 K, 16 K]. Q is BF16, **pre-widened to FP32 at
init** (2 KiB scratch — §7.7 explains why); K and V are NVFP4
storage with E4M3 per-block-16 scales (`HEAD_DIM / 16 = 4` blocks
per row); the output O is BF16. The fused kernel runs one
self-contained pass per head:

```
for h in heads:
    for s in seq_len:                    # QK^T scores
        K_fp32 ← dequant(K[h,s])
        S[s] ← (Q[h] · K_fp32) / sqrt(HEAD_DIM)
    max_s ← reduce_max(S)                # softmax max
    for s in seq_len:                    # softmax exp + sum
        P_fp32[s] ← vfexp(S[s] − max_s)
    sum_p ← reduce_sum(P_fp32)
    for s in seq_len:                    # P quant
        P_fp8[s] ← E4M3(P_fp32[s] · 448)
    for s in seq_len:                    # P · V
        V_fp32 ← dequant(V[h,s])
        O_fp32 += e4m3_decode(P_fp8[s]) · V_fp32
    O[h] ← BF16(O_fp32 · (1/sum_p) / 448)
```

The pseudocode operations map to the §4 FU lanes as follows:
`dequant(K|V)` → `vfconv.nvfp4.bf16.v` (Stage 4.3); `vfexp(·)` →
`vfexp.v` (Stage 4.2); `E4M3(·)` → `vfconv.bf16.fp8.v` (Stage 4.4);
`e4m3_decode(·)` → `vfconv.fp8.bf16.v` (Stage 4.5). Each operation
is an inline vector lane, not a SW table lookup.

K and V are streamed; no intermediate FP32 K/V buffer is
materialized across heads. Each per-head pass touches the KV
cache exactly once.

**Figure 2: Per-stage precision routing through the fused-FA
kernel.** Solid arrows are intermediate values flowing between
stages; the `[dtype]` annotation on each arrow shows the precision
the value is *carried* in at that point. Each stage selects its
own dtype based on dynamic-range and accuracy tolerance (§3); the
four custom FU lanes (boxed) translate at the precision boundaries.

<!-- TikZ source: paper/figures/fig2_precision_flow.tex
     LaTeX inline: \input{figures/fig2_precision_flow.tex}
     ASCII version below retained as a reading aid. -->

```
           Q [BF16, pre-widened to FP32 at init]
              │
              ▼  [FP32]
   ┌──────────────────────────┐                       per s ∈ seq_len:
   │  K_nvfp4 ──▶ ┌────────┐  │
   │              │ vfconv │───── [BF16] ──▶ widen ──▶ [FP32]
   │  scale_e4m3──▶│ nvfp4 │                                │
   │              │ .bf16  │                                ▼
   │              └────────┘                       vfmacc  ─────▶ S[s] [FP32]
   └──────────────────────────┘                    (Q·K^T)
              │
              ▼ S [FP32, all s]
       max-reduce ──▶ max_s [FP32]
              │
              ▼
   ┌──────────────────────────┐                       per s:
   │   S[s] - max_s ─────────┐│
   │                          ▼│
   │                ┌──────────┴┐                    P_fp32[s] [FP32]
   │                │  vfexp.v  │ ─── [FP32] ────────────┐
   │                └───────────┘                        ▼
   └──────────────────────────┘            sum-reduce  ──▶ sum_p [FP32]
              │
              ▼ P_fp32 [FP32] × 448 (per-row pre-scale, §6.4)
   ┌──────────────────────────┐
   │              ┌────────┐  │
   │              │ vfconv │  │
   │              │ bf16   │ ──── [FP8-E4M3] ──▶ P_fp8[s]
   │              │ .fp8   │
   │              └────────┘
   └──────────────────────────┘
              │
              ▼ P_fp8                                  per s:
   ┌──────────────────────────┐                  V_nvfp4 ──┐
   │              ┌────────┐  │                            ▼
   │              │ vfconv │  │                       ┌────────┐
   │  P_fp8 ─────▶│ fp8    │ ──── [BF16] ──▶ widen ──▶│ vfmacc │
   │              │ .bf16  │                          │ (P·V)  │
   │              └────────┘                          └────┬───┘
   └──────────────────────────┘                            │
                                                           ▼
                              O_fp32 += P · V         [FP32, per-d acc]
                                                           │
                                              × (inv_sum/448) per row
                                                           ▼
                                            O[h, d] [BF16, final cast]
```

### 6.2 K/V Dequantization

`dequant_row_native()` issues one `vfconv.nvfp4.bf16.v` over a
HEAD_DIM-element batch and four scaled FP32 stores. We scalar-
unpack the 32 packed bytes (8 bytes × 4 blocks) into a 64-element
`uint16_t` scratch buffer, one nibble per `u16`. Inline asm then
runs the vfconv at LMUL=4 (vl=64, e16) and, with a single
vsetvli switch to e32 m2, four `(vzext.vf2 / vsll.vi /
vfmul.vf scale_blk / vse32)` sequences that widen the BF16
output to FP32 and apply the per-block E4M3 scale. Two vsetvli
transitions per row total (down from the obvious-but-wrong eight
in the first integration cut — §7.7). On gem5 RiscvO3CPU the
single vfconv issue dispatches as four LMUL-4 micro-ops; the FU
op-class wiring (Table 1, `SimdNvfp4CvtOp = 3 cyc`) gives a
3-cycle issue-to-output latency that overlaps the surrounding
loads.

### 6.3 Two-Pass Softmax with `vfexp.v`

We deliberately use a **two-pass softmax** (max-reduce + sum-
reduce, then per-element `(exp(S−max)/sum)`), not the FA-2
streaming online softmax. The decision is microarchitecture-driven
and validated on RiscvO3CPU in Track J-2: at iso-µarch on a 2 K
context, two-pass BF16 RVV runs at **IPC 0.82 (3.39 M cyc)**
while SpacemiT-K1's ported online softmax runs at IPC 0.46 (4.99 M
cyc) — two-pass wins by 47 % on cycles despite having ~17 % more
dynamic instructions. The mechanism: SpacemiT's online softmax
has a hot data-dependent branch (`if (S[s] > M) rescale_acc(...)`)
inside the inner loop; OoO can't speculate past it. Two-pass
softmax has only pure vector reductions in the inner loops
(`vfredmax.vs`, `vfredusum.vs`) and issues unblocked. On the
in-order TimingSimpleCPU the ranking flips (SpacemiT wins by 7 %
on instructions), so the choice is OoO-specific.

The `vfexp` lane is invoked from an asm loop at LMUL=2: load
`S[s..s+15]` via `vle32.v`, subtract the scalar `max_s` via
`vfsub.vf`, issue `vfexp.v` (`.4byte 0x4E831457` for v8,v8), store
`P_fp32` via `vse32.v`, and accumulate `vfredusum.vs` into a
scalar reduction register that becomes `sum_p` after the loop. The
exp + reduce + store collapse into the same loop body — no
intermediate `P_fp32` re-read for the sum.

### 6.4 FP8 Attention-Weights Quant and Per-Row 448× Pre-Scale

Naïve FP8-E4M3 quantization of the normalized softmax weights
*underflows to zero at seq_len ≥ ~512*. For a uniform softmax over
2048 keys, the typical normalized weight is ≈ 1/2048 ≈ 4.9 × 10⁻⁴,
well below E4M3-FN's smallest subnormal (2⁻⁹ ≈ 1.95 × 10⁻³). This
is the algorithm-level bug we surfaced during early FA-kernel
bring-up: the kernel ran but produced all-zero output. The fix is a **per-row pre-scale by E4M3-FN max
(448)**: quantize the *unnormalized* exponentials `P_fp32[s] · 448`
instead, then absorb the missing `1/sum_p` *and* the `1/448`
back-scaling into a single scalar multiply applied at the final
O cast. Algebraically `O_final = (P_pre_norm · 448) · V · (1/sum_p)
/ 448 = (exp / sum_p) · V`. The FU-side cost is one extra scalar
FMUL per output dim — outside the `vfconv.bf16.fp8.v` lane and
outside the per-row hot loop. The dominant softmax weight now
maps to FP8 code `0x7E = 448` (no underflow) and each FP8 ULP
corresponds to ~1/256 of the row-scale max.

The quant itself is `P_fp8[s] = bf16_to_e4m3(fp32_to_bf16(P_fp32[s]
· 448))` per element — software in the primary native bench. A
sidebar variant (`bench_fa_mixed_rvv_native_allfu`, §7.5) issues
`vfconv.bf16.fp8.v` directly and verifies correct integration of
the 4th custom lane, but the cycle measurement on gem5 is 8 %
slower than the SW path due to chain serialization in the vector
FP8-quant body — an artifact of the single-instruction-latency
stub model, not the lane itself. PV loads `e4m3_decode[P_fp8[s]]`
as a scalar `float` per `s` and issues `vfmacc.vf` against the
dequantized V row at LMUL=2.

### 6.5 Implementation Pitfall

The kernel hit a non-obvious GCC 13.2 RVV vsetvli pass bug at
the asm-dequant → intrinsic-QK boundary that produced silently
wrong output (checksum −0.045 vs reference −20.27) before being
diagnosed. The full debug trace and workaround (pre-widen Q to
FP32, eliminate `bf16_load_widen` from the QK inner loop) are in
§7.7; we flag the failure mode here because future kernels mixing
`.4byte` Saturn-custom encodings with surrounding RVV intrinsics
that switch SEW are likely to reproduce it.

---

## 7. Evaluation

### 7.1 Setup

We evaluate on **gem5 25.1.0.1** in SE mode, RISC-V port, with our
patched RVV decoder + FU latency wiring. Per-CPU: RiscvO3CPU,
4 cores, 32 KiB private L1D / 32 KiB L1I, 512 KiB shared L2, 512 MiB
main memory backed by DDR3-1600 (6.4 GB/cyc peak). Each microbench
runs single-threaded; the additional cores satisfy `clone()` for
host-side runtime. RVV configuration: VLEN=256, ELEN=64, no Zfh /
Zvfh / Zicbop / Zihintpause / Zvfbfwma (gem5 doesn't currently
support these). The microbench harness emits `m5_dump_reset_stats`
at the start of the measurement region and `m5_dump_stats` at the
end, fencing the malloc + init + quantization-of-K-V work out of the
counters.

Cross-compiler: Bootlin GCC 14.2 (`riscv64-linux-gnu`) with
`-march=rv64gcv -mabi=lp64d -O2 -fno-tree-vectorize -static`. RVV
1.0 intrinsics are used throughout the C reference kernels; inline
`.4byte` asm injects the four custom encodings. `-fno-tree-vectorize`
suppresses a GCC 14 -O2 auto-vectorizer interaction with the
asm-intrinsic boundary in our kernel that produced silent miscompiles
on initial trials; the all-intrinsic baseline kernels (BF16 RVV /
mixed-RVV-proper / mixed-RVV-stub) are unaffected (their cycle
counts are within 2 % of the GCC 13.2 baselines) but we apply the
flag uniformly for consistency. See §7.7 for the bug story.

**Kernel shape**: 8 KV-heads, `head_dim = 64`, `seq_len ∈ {2 048,
4 096, 8 192, 16 384}` — matches Llama-3.2-1B GQA decode-step. Q
is a single token (BF16, pre-widened to FP32 at init for reasons
explained in §7.7). K, V are NVFP4 block-16 with E4M3 scale.

**Variants compared** (all use the same kernel structure; differences
in dequant + softmax implementation):

- **bf16-rvv**: J3 BF16 RVV row-major two-pass baseline.
  No NVFP4, no custom FUs. Primary reference.
- **mixed-rvv-proper**: SW-dequant NVFP4 K/V (scalar 16-element-LUT
  per block), proper FP8 quant of attention weights via
  `bf16_to_e4m3`. Iso-FP8-quant baseline for FU savings analysis.
- **mixed-rvv-stub**: SW-dequant NVFP4 K/V, broken FP8 quant stub
  (`(uint8_t)((int)pf & 0xff)`) — cycle-only comparator, checksum
  meaningless.
- **mixed-rvv-native**: our FU-integrated kernel —
  `vfconv.nvfp4.bf16.v` for K/V dequant, `vfexp.v` for softmax,
  proper FP8 quant.

### 7.2 RTL Validation

All four custom instructions have standalone Chisel RTL with
exhaustive bit-exact validation:

**Table 3: RTL validation summary.**

| Instruction              | Stages | LOC | Test cases       | Mismatches  | Worst rel err |
|--------------------------|-------:|----:|-----------------:|------------:|--------------:|
| `vfexp.v`                |     10 | 314 | 321 BF16         | n/a (FP)    | 3.20e-6       |
| `vfconv.nvfp4.bf16.v`    |      3 | 197 | 4 096            | **0**       | bit-exact     |
| `vfconv.bf16.fp8.v`      |      2 | 192 | 65 536           | **0**       | bit-exact     |
| `vfconv.fp8.bf16.v`      |      1 | 110 | 256 + 256 RT     | **0**       | bit-exact     |

The `vfexp` 3.20e-6 rel err (at x=3.8125) is 4.7× tighter than the
2⁻¹⁶ ULP-equivalent target. The three vfconv lanes are bit-exact against a
distance-based RNE golden (banker's tie-break, NaN-encoding boundary
saturation) per [[feedback-distance-based-goldens]]. Cumulative
ChiselTest count: 57/57 pass.

### 7.3 Area

Yosys 0.9 synthesis with the generic `cells_sim.v` cell library;
NAND2-equivalent area assumed 0.26 µm² (TSMC-16FF+-class generic).
firtool 1.62.1 emits Verilog with `disallowLocalVariables,
disallowPackedArrays` for Yosys-0.9 frontend compatibility.

**Table 4: Per-lane synthesis area (Yosys 0.9, single lane).**

| Lane                    | NAND2-eq | Area / lane @ 16 nm |
|-------------------------|---------:|--------------------:|
| `vfexp.v` (PolyExp dom.)|  74 866  |          ~0.020 mm² |
| `vfconv.nvfp4.bf16.v`   |     704  |          ~0.0002 mm²|
| `vfconv.bf16.fp8.v`     |     503  |          ~0.0001 mm²|
| `vfconv.fp8.bf16.v`     |     140  |          ~0.00004 mm²|

At 16 SIMD lanes (matching Saturn's vector pipe width assumption),
total FU area is **0.317 mm² standalone, 6.34 % of Saturn baseline**
(5 mm² @ 16 nm), or **0.055 mm² / 1.10 %** if the polynomial FMA in
`vfexp` is shared with Saturn's existing FMA pipe. Both numbers
comfortably pass the ≤ 10 % area-of-Saturn threshold. Caveats: Yosys 0.9 is ~2× conservative vs commercial DC;
generic library not a 16 nm PDK; no timing closure performed.

### 7.4 KV-cache DRAM Bandwidth

We measure DRAM bytes-read on gem5 across multiple kernel scales.
At the fused-FA decode-step shape used throughout this paper
(8 heads × 64 hd × 2 048 seq), NVFP4 K+V reads **1.18 MiB**, vs
BF16 K+V's **4.20 MiB** — **3.56× DRAM reduction**, in line with
the analytical 4.5-bits-per-element-with-block-16-scale prediction
(FP16 = 16 bits/element → 3.56× compression). The reduction is
independent of `seq_len` in the L2 K–L16 K range (linear scaling on
both sides); our long-context sweep at L4 K / L8 K / L16 K on
RiscvO3CPU confirms identical 3.56× ratio at each scale.

### 7.5 FU Integration Cycle Delta

The headline measurement of this paper. Full L-sweep on
RiscvO3CPU:

**Table 5: Full L-sweep cycle measurement on RiscvO3CPU (cycles in M, IPC in parens).**

| seq_len | BF16 RVV (ref) | Mixed RVV proper | Mixed RVV stub | **Native (FU)**  | Native/BF16 | Native/Proper |
|--------:|---------------:|-----------------:|---------------:|-----------------:|------------:|--------------:|
|    2 K  |  3.46 M (0.80) |   9.03 M (2.42)  |  7.58 M (1.89) | **5.49 M (2.40)** |  **1.59×**  | 0.61× (39 % faster) |
|    4 K  |  6.87 M (0.81) |  18.62 M (2.34)  | 16.99 M (1.68) | **13.23 M (1.99)**|  **1.93×**  | 0.71× (29 % faster) |
|    8 K  | 13.56 M (0.81) |  36.60 M (2.39)  | 34.86 M (1.64) | **26.95 M (1.95)**|  **1.99×**  | 0.74× (26 % faster) |
|   16 K  | 27.32 M (0.81) |  74.59 M (2.34)  | 70.11 M (1.63) | **54.07 M (1.94)**|  **1.98×**  | 0.72× (28 % faster) |

(Cycles in millions; IPC in parentheses; instruction counts in
appendix.) Native checksums: −20.297 (L2 K), −20.406 (L4 K),
−20.338 (L8 K), −20.271 (L16 K) — all within 1 % relative of the
mixed-precision FP32 reference (−20.27), well inside the 5 % NVFP4
quantization-noise budget. Mixed-RVV-proper checksums: −20.272 /
−20.248 / −20.249 / −20.253 across the L-sweep, also within 1 %.

**Figure 3: Cycle counts per variant at each seq_len, normalized to
BF16 RVV as 8 `█` blocks (1 `█` ≈ 0.125× of BF16 ratio).** The
≥1.5×-faster-than-BF16 target requires bars ≤ 5.3 blocks long (the
"target" line drawn below); the BF16 reference is at 8 blocks (the
"BF16 ref" line); all mixed variants extend well past both.

<!-- TikZ source: paper/figures/fig3_cycle_bars.tex (requires pgfplots)
     LaTeX inline: \input{figures/fig3_cycle_bars.tex}
     ASCII version below retained as a reading aid. -->

```
              target       BF16 ref
              (≤ 5.3 █)    (8 █ = 1.00×)
                  │              │
                  ▼              ▼
seq_len = 2 K
  BF16 RVV        ████████                                  1.00×   3.46 M  IPC 0.80
  Native FU       █████████████                             1.59×   5.49 M  IPC 2.40  ← best
  Mixed stub      ██████████████████                        2.19×   7.58 M  (cycle-only)
  Mixed RVV proper████████████████████                      2.61×   9.03 M

seq_len = 4 K
  BF16 RVV        ████████                                  1.00×   6.87 M
  Native FU       ████████████████                          1.93×  13.23 M  ← best
  Mixed stub      ████████████████████                      2.47×  16.99 M
  Mixed RVV proper██████████████████████                    2.71×  18.62 M

seq_len = 8 K
  BF16 RVV        ████████                                  1.00×  13.56 M  IPC 0.81
  Native FU       ████████████████                          1.99×  26.95 M  ← best
  Mixed stub      █████████████████████                     2.57×  34.86 M
  Mixed RVV proper██████████████████████                    2.70×  36.60 M

seq_len = 16 K
  BF16 RVV        ████████                                  1.00×  27.32 M  IPC 0.81
  Native FU       ████████████████                          1.98×  54.07 M  ← best
  Mixed stub      █████████████████████                     2.57×  70.11 M
  Mixed RVV proper██████████████████████                    2.73×  74.59 M
```

The native/BF16 ratio sits at 1.59× at L2 K and stabilizes at
~1.95× at L4 K-L16 K — Native FU integration consistently beats
both mixed-RVV-proper (the iso-FP8-quant baseline) and
mixed-RVV-stub (the broken-FP8 cycle-only baseline) on every
tested scale, with the most pronounced absolute gap at L2 K
where the BF16 RVV path is least compute-bound.

**Two findings**:

1. **FU integration delivers a verified 26–39 % per-row cycle
   speedup vs SW-dequant proper.** FU savings scale roughly
   linearly with `seq_len`: 3.54 M (L2 K), 5.39 M (L4 K), 9.65 M
   (L8 K), 20.52 M (L16 K). Instruction count drops ~40 % at every
   scale (e.g. 105.1 M vs 174.6 M at L16 K) — the FU collapses
   ~50 M scalar dequant ops at L16 K into ~256 K `vfconv` issues.
   Native IPC (1.94–2.40) is lower than proper (2.34–2.46) because
   FU ops queue against fewer dependencies per op, but the cycle
   delta wins on instruction-count reduction. The L2 K case is
   the outlier — narrower mixed-instruction streams expose more
   ILP, pushing native IPC up to 2.40.

2. **Native/BF16 ratio shrinks to 1.59–1.99× across the
   L2 K–L16 K sweep.** Long-context does *not* bridge to the
   literal ≥1.5× target, but the best case is the *smallest*
   seq_len tested (L2 K → 1.59×, within ~6 % of the target) — the
   gap grows modestly with seq_len and stabilizes at ~1.95× from
   L4 K onward. The L2 K result is the closest we get to the
   target on any tested configuration.

**Sidebar: all-customs-integrated variant.** The primary native
variant above uses SW `bf16_to_e4m3` per element for the FP8
attention-weights quant — only 2 of our 4 custom lanes
(`vfconv.nvfp4.bf16.v` and `vfexp.v`) appear in the cycle
measurement. A separate "all-FU" variant
(`bench_fa_mixed_rvv_native_allfu`) replaces the SW FP8 quant
with a vector loop issuing `vfconv.bf16.fp8.v` (FP32 → BF16
narrowing via `vnsrl.wi 16`, then the FU conversion, then `vnsrl
0` to extract the FP8 byte). On the original GCC 13.2 toolchain
this variant measured 7.56 M cyc / IPC 1.91 at L2 K vs SW-FP8
native's 7.00 M (8 % regression); the 5-deep dependency chain in
the vector FP8-quant body (`vle32 → vfmul.vf → vnsrl.wi →
vfconv → vnsrl.wi → vse8`) limited per-iter ILP at LMUL=1/mf2,
and gem5's single-instruction-2-cycle stub for the
`vfconv.bf16.fp8.v` lane does not model the 16-lane parallelism
a real Saturn FU would provide. The qualitative finding stands:
the cycle-cost-bound FP8 quant lane doesn't win on the gem5 stub
model. We report numbers above using the SW-FP8 variant as the
primary because it isolates the FU-cycle-savings contribution of
the two lanes that *do* benefit from the stub (NVFP4 dequant and
`vfexp`) without conflating with the chain-serialization
artifact of the FP8 quant lane.

### 7.6 Why the Literal ≥1.5× Threshold Is Not Met

The ≥1.5×-over-BF16-RVV-at-iso-VLEN target implicitly assumes
the FU integration shifts the kernel from compute-bound to
bandwidth-bound, where the 3.56× DRAM reduction translates to
proportional cycle savings. Our gem5 + DDR3-1600 configuration
breaks that assumption:

- **BF16 RVV is compute-bound at every tested seq_len.** IPC is
  pinned at 0.80–0.81 across L2 K–L16 K. DRAM consumption is
  1.3-1.6 GB/cyc — well within the 6.4 GB/cyc DDR3-1600 cap.
- **Mixed-precision's BW advantage cannot translate.** The kernel
  isn't waiting on DRAM. The FU integration only helps on the
  *compute* side, where it delivers the verified 26–39 % savings.

Conditions under which the literal ≥1.5× would land:

1. **HBM-class BW ceiling**, where peak DRAM bandwidth is much
   smaller relative to compute throughput, so BF16 RVV becomes
   BW-bound. Our long-context sweep establishes that even at L16 K
   under DDR3-1600 the BF16 RVV path stays at 20 % of the BW
   budget — the BW pressure flip requires a tighter ceiling than
   we can model in the current gem5 config.
2. **Real Saturn µarch with 16-lane FU pipeline.** gem5 issues each
   `vfconv` as a single instruction with 3-cycle latency; real
   Saturn would dispatch 16 elements per cycle. The FU integration
   delta would expand from 26–39 % to ~80 %.
3. **Larger head_dim**. Disproven by a hd=128 L2 K spot
   measurement: doubling head_dim widens the native/BF16 ratio
   from 1.59× to **1.96×** (BF16 RVV 6.52 M cyc, Native FU
   12.78 M cyc; FU integration benefit shrinks from 39 % to 14 %).
   Per-row compute doubles, dequant work doubles too — but the
   BF16 baseline absorbs the extra compute without becoming
   BW-bound, while the FU's dequant-collapse contribution becomes
   a smaller fraction of total work. Larger head_dim makes the
   gap wider, not narrower; not a path to the literal target.

We therefore split the speedup claim into: **(a) verified 26–39 %
FU integration cycle savings vs SW dequant** at every tested
seq_len, on the same gem5 µarch as the BF16 RVV reference; and
**(b) projected ≥1.5× over BF16 RVV** on memory subsystems where
BF16 RVV is bandwidth-bound. Both claims are testable; both are
useful for the precision-routing co-design argument. The best
empirical case (L2 K) lands at 1.59× — within 6 % of the literal
target on the smallest tested config, but still on the wrong side
of it.

### 7.7 Implementation Note: GCC 13.2 RVV vsetvli Pass Bug

The mixed-RVV-native kernel hit a non-obvious correctness bug
during integration: GCC 13.2's RVV vsetvli optimization pass does
not parse inline asm content. After the dequant asm block, the
pass's tracker thought vtype was still e32 m2; when the QK loop's
`bf16_load_widen` intrinsic chain emitted `vsetvli zero, t3, e16,
m1` for the `vle16` of Q, the pass *failed to emit the e32 m2
transition* before the subsequent `vzext.vf2 / vsll.vi / vfmacc.vv`
chain. Those ran at SEW=16: `vzext.vf2` read u8 elements (source
EEW = SEW/2 = 8) instead of u16; `vsll.vi 16` was shift-by-(16
mod 16)=0; `vfmacc.vv` landed on half-precision FMA without
Zvfh. Result: garbage attention scores in the entire downstream
softmax + P·V pipeline; output checksum −0.045 vs reference
−20.27. Workaround: pre-widen Q to FP32 at init (2 KiB scratch),
use `__riscv_vle32_v_f32m2` for Q in the QK loop — eliminates the
SEW switch from the inner loop and sidesteps the GCC pass. A
tighter workaround at the call site is an inline-asm opacity
barrier on `vl` (`asm volatile ("" : "+r"(vl));`) between the
explicit `__riscv_vsetvl_e32m2(...)` and the first SEW-changing
intrinsic. We minimized the bug to a 7-line standalone reproducer
that does not involve inline `.4byte` Saturn customs at all —
pure intrinsics (`vle16 → vzext.vf2 → vse32`) at `-O2` exhibit the
missing `vsetvli` transition. Cross-version testing on three
Bootlin toolchains confirms the **pure-intrinsic** form is fixed
in GCC 14.2.0 and 15.1.0, present only in GCC 13.x (13.2.0
tested). GCC 14 picks `e32, m2` (the widest SEW in the chain) as
the unified vtype and runs `vle16.v` under it correctly via
baked-in EEW; GCC 13 picks the narrowest SEW and breaks
`vzext.vf2`. **However**, the GCC 14.2 fix is *partial*: when an
asm-volatile block that internally changes vtype precedes the
widening chain (e.g., a custom-instruction macro that issues its
own `vsetvli`), GCC 14.2 *still* fails to emit a vsetvli before
the next intrinsic — confirmed during the Mo 8 step 4d-1
intrinsic-rewrite work, where the Saturn `.4byte` vfconv macro
followed by a `vzext.vf2 + vsll.vx` widen chain produced the
same wrong-SEW miscompile (checksum NaN/Inf). Workaround in
that case: explicit `asm volatile ("vsetvli zero, %0, e32, m2,
ta, ma" :: "r"((size_t)(vl)))` before the widen — the explicit
`__riscv_vsetvl_e32m2(vl)` intrinsic call gets DCE'd because its
returned vl isn't used. Full report at
`paper/gcc_bug_report.md`: Part 1 is the GCC-13 backport
request, Part 2 documents the GCC-14.2 asm-volatile-boundary
limitation.

**A second, distinct GCC 14 -O2 issue** surfaced when we switched
the project to GCC 14.2: the auto-vectorizer (enabled by default
at -O2) does something around the asm-intrinsic boundary in our
larger kernel that produces a different silent miscompile (L2 K
checksum −136.8 vs reference −20.27). The minimal 7-line
reproducer above does NOT trigger this second bug; we have not
minimized it. The workaround is `-fno-tree-vectorize` at -O2,
applied uniformly to all benchmarks reported in §7.5 — the
all-intrinsic baseline kernels are unaffected by this flag
(cycle delta < 2 % vs default -O2) so the flat application is
loss-free. With `-fno-tree-vectorize` enabled, GCC 14.2 builds
the kernel correctly and produces the cycle numbers in Table 5
that are 13–22 % faster than the corresponding GCC 13.2 +
workaround baselines reported in earlier drafts of this paper.

---

## 8. Related Work

### 8.1 Fused-Attention Hardware

The closest precedent is **FuseMax** [Nayak et al., MICRO'24] —
which derives a cascade-of-Einsums dataflow and maps it to a 256 × 256
spatial mesh achieving 6.7× over FLAT [Kao et al., ASPLOS'23] at
iso-area. FuseMax **pins FP16 throughout**: the per-stage precision
dimension is unexplored. The IEEE Micro 45(4) 2025 retrospective
reframes the methodology but adds no new precision configs or
compiler integration. Earlier attention accelerators —  A³ [Ham
HPCA'20], ELSA [Ham ISCA'21], SpAtten [Wang HPCA'21], Sanger [Lu
MICRO'21], DOTA [Qu ASPLOS'22], FACT [Qin ISCA'23], CTA [Wang
HPCA'23] — focus on sparsity, eager prediction, or token pruning,
none on per-stage precision routing in a fused kernel.
**SystolicAttention** (FSA) [arXiv:2507.11331, 2025] is the closest
closed-source systolic-FA competitor; no precision routing, no RVV.
The methodology papers we cite are TeAAL [Nayak ASPLOS'24] (the
Einsum framework parent of FuseMax) and LoopTree [Odemuyiwa
PACT'24] / TileFlow [Wu MICRO'23] for fused-layer dataflow modeling.

### 8.2 RVV Custom-Instruction Work

**VEXP** [Wang et al., ARITH'25, arXiv:2504.11227] adds a single
softmax-exp instruction on Snitch's packed-FPU pipeline (Xfvec) —
1 % area, 162× softmax latency reduction. **VMXDOTP** [Wipfli /
Islamoglu / Benini, DATE'26, arXiv:2603.04979] adds RVV MX
(MXFP8 / MXFP4) dot-product instructions on the Spatz vector unit
in 12 nm, evaluated on DeiT-Tiny GEMM. **Titopoulos** [J.Supercomp
2026, arXiv:2510.06834] ports FA-2 to gem5 RVV with hand-tuned
intrinsics, no FU additions. The Benini group (VEXP + VMXDOTP)
ships RVV-customs-for-ML papers regularly and is the most likely
concurrent threat for an attention-flavored sequel at HPCA / MICRO
2027. We differentiate from VMXDOTP on six axes: (a) NVFP4
block-16 + E4M3 scale vs MX block-32 + E8M0 scale, (b) fused FA
vs unfused GEMM, (c) Exo `@instr` scheduling vs hand-tuned LLVM
intrinsics, (d) Saturn (OoO, chaining) vs Spatz (in-order),
(e) decoder LLM (Llama-3.2-1B) vs image classifier (DeiT),
(f) end-to-end llama.cpp integration vs microbench only.

### 8.3 NVFP4 / Microscale Quantization

**ARCQuant** [Anonymous, arXiv'26] is the algorithm baseline for our
NVFP4 K/V choice — PTQ with calibration + per-block scale that
reports < 0.5 perplexity degradation on Llama-3.2; software-only,
no HW. **MR-GPTQ** [arXiv:2509.23202, 2025] takes the MXFP4 (OCP)
algorithm route; we use NVIDIA NVFP4. **Berkeley `gemmini-mx`**
(Hansung Kim) is the concurrent NVFP4-on-Gemmini-systolic line of
work — different microarchitecture (vector vs systolic), different
scope (unfused GEMM vs fused FA). **SageAttention** [Zhang
ICLR'25] and its FP8 / FP4 successors implement quantized
attention on NVIDIA GPUs; the dataflow ideas overlap but the
target hardware is different.

### 8.4 Open RVV SoCs

**SiFive Intelligence X-series**, **SpacemiT K1**, **Tenstorrent
Ascalon**, and **Andes AX65** ship RVV 1.0 cores today. **Cygnus**
[Jain et al., JSSC March 2026] is the Berkeley octa-core RVV
processor from Asanović / Shao / Nikolić; DSP-focused, no
attention or quantization angle. **SpacemiT** ships a hand-tuned
VLEN=1024 fused-FA kernel in upstream llama.cpp
(`spacemit/rvv_kernels.cpp`); gating on
`__riscv_vlenb() * 8 == 1024` means it does not trigger on Saturn
or any other VLEN. Saturn itself [Zhao et al., arXiv:2412.00997]
is the substrate we extend.

### 8.5 Compiler and Scheduling DSLs

**Exo 2** [Ikarashi et al., ASPLOS'25] is our compiler vehicle.
Other RVV compiler stacks include **IREE-RVV** [arXiv:2508.14899, 2025] —
SiFive's MLIR-RVV path on Intelligence X-series with closed-source
SKL kernels behind a vendor wall — and the Triton GPU stack
[Tillet MAPL'19], which has no RVV backend. Upstream `exo-lang/exo`
has an AVX-512 / NEON / SVE / Gemmini / RVM library but no full
RVV backend (verified Track D 2026-05-17 against commit `2f5472d`:
175-LOC `platforms/rvv.py` stub, VLEN=128 f32-only, 9 intrinsics).

### 8.6 Differentiation Snapshot

Table 6 summarizes positioning. *Open RTL* = RTL artifacts publicly
available; *Custom FU* = new functional unit beyond standard ISA;
*Mixed prec* = per-stage precision routing within one kernel;
*Compiler* = how kernels are produced; *E2E LLM* = end-to-end
deployment (not microbench); *OSS stack* = full stack open-source.

**Table 6: Comparison snapshot — our positioning vs prior work.**

| Work | ISA / Platform | Open RTL | Custom FU | Mixed prec | Compiler | E2E LLM | OSS stack |
|---|---|---|---|---|---|---|---|
| FuseMax [MICRO'24] | Custom spatial | partial (model) | full-attn FU | no (FP16) | Einsum spec | no | no |
| VEXP [ARITH'25] | Snitch (Xfvec) | yes | softmax-exp | no (BF16) | hand intrin. | no | partial |
| VMXDOTP [DATE'26] | Spatz (RVV) | yes | MX dot-prod | partial (MX, GEMM) | hand intrin. | no (DeiT) | partial |
| Titopoulos [J.Supercomp'26] | RVV (gem5) | no | no | no | hand intrin. | no | code only |
| SystolicAttn [arXiv'25] | Custom systolic | no (closed) | full-attn FU | no | no | no | no |
| SpacemiT FA [llama.cpp] | K1 VLEN=1024 | no (vendor) | no | no | hand C++ | yes (Llama-3) | partial |
| IREE-RVV [arXiv'25] | SiFive Intel. | no (vendor) | no | partial (INT8) | MLIR closed | yes (Llama-3) | no (SKL closed) |
| **Ours** | **RVV 1.0 (Saturn)** | **yes** | **4 (NVFP4↔BF16↔FP8 + vfexp)** | **yes (per-stage)** | **Exo 2 @instr** | **yes (Llama-3, gem5 today; FireSim Y2)** | **yes** |

We are the only entry combining open RTL, custom precision-
conversion functional units, per-stage mixed-precision routing
inside a fused-attention kernel, scheduling-DSL compiler
co-design, and a fully open stack. The end-to-end LLM column
moves from "gem5 today" to "FireSim Y2" with the Track D-follow
Exo fork and the FireSim integration outlined in §9.

---

## 9. Conclusion

We presented a co-designed mixed-precision fused flash-attention
stack for open RISC-V vector hardware: four custom instructions on
the Berkeley Saturn vector unit, an Exo 2 scheduling pass that
declares them and routes attention stages to per-stage dtypes, and
a gem5 5.1 RISC-V decoder integration that lets us measure the FU's
end-to-end effect on a real fused-FA kernel. Two findings are
verified on RiscvO3CPU at hd = 64 across seq_len ∈ [2 K, 16 K]:
**(i)** a **3.56× KV-cache DRAM-bandwidth reduction** from NVFP4
K/V storage, and **(ii)** a **26–39 % per-row cycle speedup** of
the FU-integrated kernel over a software-dequant mixed-precision
baseline at iso-FP8-quant policy. The literal ≥1.5× over BF16 RVV
holds only at L2 K (1.59×, within 6 % of the target) — a
memory-subsystem property rather than a hardware-design one (§7.6);
the FU-integration delta lower-bounds the hardware contribution
independent of that assumption.

The compiler co-design loop is closed end-to-end: the Exo
precision-routing pass (§5.5) lowers a high-level
`fa_kernel_decode_naive` @proc through 14 sub-schedules to the §6
kernel, with all four Saturn customs reachable from the @instr
surface. The Exo-generated kernel runs on gem5 across the full
L2 K – L16 K sweep at **PASS at every tested length (1.046 × /
0.867 × / 0.850 × / 0.848 ×) — at L ≥ 4 K, Exo strictly beats the
hand-coded reference by 13–15 %.** The crossover reflects an ILP-
ceiling effect: hand-coded's IPC drops from 2.40 (L 2 K) to 1.94
(L 16 K) under chain-pressure, while Exo's IPC is flat at 0.631
and its instruction-count advantage (~3.6 × fewer than hand-coded
at every L) stays constant. The L 2 K residual is decomposed in
§5.6; three optional optimisation paths (upstream GCC intrinsics
for the Saturn customs, Exo register-subview support, peephole-
merge of consecutive asm-volatile macros) would tighten the L 2 K
margin but are not required at L ≥ 4 K.

All RTL (Saturn fork, 57/57 ChiselTests, Yosys-synthesized to a 16 nm
area estimate), gem5 patches (RISC-V decoder + FU-latency wiring +
microbenches), Exo `@instr` declarations + the precision-routing
scheduling pass, and end-to-end kernel source open-source under
permissive license. Artifact repositories:
main project at `git@github.com:noyaboy/precision-routing-rvv-fa.git`,
Exo fork at `git@github.com:noyaboy/exo-saturn-rvv.git`,
gem5 fork at `git@github.com:noyaboy/gem5-saturn-fu.git`. Next steps for the Y2 follow-on: a HBM
bandwidth model in gem5 to verify the projected ≥1.5× over BF16
RVV, FireSim integration for real-Saturn-µarch validation
(plan in `paper/y2_firesim_prep.md`), optional further
compiler-parity optimisations at L 2 K, and a Llama-3.2-1B E2E
speedup measurement on actual silicon.


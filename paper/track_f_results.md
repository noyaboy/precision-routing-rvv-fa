# Track F results — `vfconv.nvfp4.bf16.v` Chisel lane (2026-05-17)

## Verdict: PASS — bit-exact dequant across all 4096 (elt, scale) codepoints

The first conversion-lane FU primitive (`vfconv.nvfp4.bf16.v`) is implemented
in standalone Chisel at `/home/noah/project/riscv/saturn-fu/`, paired with
`PolyExpQ2_30` and `VFExpLane` from Mo 3 first checkpoint. Mo 3 second
checkpoint (the second FU primitive) is closed.

**One-line summary:** A 3-stage pipeline that dequantises an NVFP4 (E2M1)
element scaled by an E4M3 block scale and emits a BF16 result, with **0 / 4096
mismatches** against an FP64-multiply-and-round-to-BF16 golden across every
(elt, scale) codepoint pair.

This eliminates the software-dequant overhead measured in Mo 2 (+21%
instructions on TimingSimpleCPU). Once this lane is wired into Saturn, the
NVFP4 K-cache cycle delta — currently +12% on a SimpleCPU because dequant runs
through software LUTs — should go negative, matching the analytical 3.77×
bandwidth win we already measured.

## Motivation

Mo 2 microbench (NVFP4 vs FP16 K-cache, gem5 TimingSimpleCPU + 32 KiB L1D +
512 KiB L2 + DDR3-1600) passed the ≥30 % bandwidth-reduction checkpoint
comfortably (73.5 % reduction, 3.77 × compression). But the cycle delta was
+12 % unfavourable for NVFP4 because the software dequant path adds +21 %
instructions per token: per-byte unpack via lookup, per-block scale multiply,
per-element scale apply, all running on the scalar core.

A real `vfconv.nvfp4.bf16.v` FU at 1 cycle/element of vector lane (16 elements
per cycle at LMUL=2, matching Saturn's vector lane parallelism) removes the
entire compute overhead. Mo 6 (≥1.5 × SpacemiT K1 FP16 FA) and Mo 10 (E2E
Llama-3.2-1B Saturn-mixed-prec vs K1 FP16) both depend on this conversion
being effectively free, so building the RTL primitive early de-risks both
checkpoints.

## Spec (recap from `fu_sketch.md` + `precision_config.md`)

- **Input**: NVFP4 element (4-bit E2M1, OCP spec) + E4M3 block scale
  (8-bit, NVIDIA NVFP4 / OCP FN variant with `exp=15 & m=7` reserved as the
  only NaN — no Inf encoding).
- **Output**: BF16.
- **Latency**: 3 cycles (sketch target was 3).
- **Throughput**: 1 elt/cycle sustained.
- **NaN propagation**: scale=NaN → output is sign-preserving QNaN
  (`Cat(elt_sign XOR scale_sign, 0xFF, 0x40)` = sign | `0x7FC0`).
- **Zero propagation**: elt=±0 or scale=±0 → output is signed zero with
  sign = elt_sign XOR scale_sign.
- **Subnormal handling**: E4M3 subnormal scales (`exp=0, m∈{1..7}`) decode to
  normal BF16 (value range `[2^-9, 7·2^-9]` is well within BF16 normal range
  down to `2^-126`). NVFP4 has only ±0.5 as subnormal element, encoded as
  normal BF16 `±0x3F00`.

## Microarchitecture

3 register stages, per `fu_sketch.md`'s 3-cycle Cycle-1/2/3 outline:

```
  Stage 1 (decode)         Stage 2 (multiply)        Stage 3 (normalize)
  ───────────────────      ────────────────────       ───────────────────────────
  NVFP4 -> BF16 (16-LUT)   8x8 mantissa multiply      Mux(prod[15], prod, prod<<1)
  E4M3  -> BF16 (3-case    a_exp+b_exp-127 SInt(10)   RNE on bits[7:0]
       combinational:      sign XOR                   exp clamp [1, 254]
       normal/subn/NaN/0)  override = a=0 or b=0      apply zero/NaN override
                              or b=NaN                pack BF16
  Reg: bf16_a, bf16_b,     Reg: prod[15:0], exp_sum,  Reg: BF16 result
       elt_zero, sc_zero,       sign, override_valid,
       sc_nan                   override_result
```

Three details worth flagging:

1. **NVFP4 -> BF16 is a 16-entry `VecInit` LUT.** Synthesisable to either a
   combinational ROM or a small mux tree — synthesis tool's choice. Negligible
   area.

2. **E4M3 -> BF16 is combinational with three branches:** normal (BF16 exp =
   E4M3 exp + 120, BF16 mant = E4M3 mant << 4), subnormal (inline 3-bit
   priority encoder + bit-shuffle for the BF16 mantissa), and NaN/zero
   short-circuits. We pay a small priority encoder per lane rather than
   broadcasting one shared decoder for the per-block scale — Mo 4 area-budget
   work can revisit this if the per-lane subnormal logic is hot.

3. **Exp-sum needs ≥10-bit signed.** A 9-bit signed accumulator wraps at
   `a_exp + b_exp > 255` (e.g. NVFP4 +4.0 → BF16 exp 129 multiplied by E4M3
   +1.0 → BF16 exp 127 gives 256). This was the only bug found during
   bring-up: the 9-bit add wrapped to a negative number, downstream
   `exp_too_small` clamped to signed zero. Fixed by widening operands to
   SInt(10.W) before adding. Lesson reinforced for the remaining vfconv lanes.

## Verification

12 ChiselTests under `saturnfu.VFConvNvfp4Bf16LaneSpec`. All pass.

| # | Test | Result |
|---|---|---|
| 1 | 3-cycle latency | PASS |
| 2 | NVFP4(+1.0) × E4M3(+1.0) → 0x3F80 | PASS |
| 3 | NVFP4(+6.0) × E4M3(+1.0) → 0x40C0 | PASS |
| 4 | NVFP4(-3.0) × E4M3(+1.0) → 0xC040 | PASS |
| 5 | elt=+0 → output ±0 (sign from XOR) | PASS |
| 6 | elt=+0, scale negative → output -0 | PASS |
| 7 | scale=+0 → output ±0 | PASS |
| 8 | scale=+NaN → output +QNaN (0x7FC0) | PASS |
| 9 | scale=-NaN → output -QNaN (0xFFC0) | PASS |
| 10 | E4M3 subnormal scale (0x04 = 2^-7) × +1.0 → 0x3C00 | PASS |
| 11 | Stream all 16 NVFP4 codepoints × scale +1.0 | PASS |
| 12 | **Exhaustive sweep: 4096 cases, 0 mismatches** | PASS |

The exhaustive sweep covers every combination of 8-bit E4M3 scale (256 values
including 254 normal/subnormal + 2 zero + 2 NaN encodings — `0x7F` and `0xFF`)
and 4-bit NVFP4 element (16 codepoints).

Golden reference is computed in Scala:

```scala
nvfp4ToDouble(e) * e4m3ToDouble(s) -> Float -> floatToBf16Bits via RNE
```

with special cases for NaN scale and zero short-circuit matching the DUT's
override path.

**Bit-exact match across all 4096 cases.** This is the strongest correctness
property we can ask of a finite-domain converter, and it forecloses any
"close-enough but reviewer-baitable" failure mode.

## Area / LOC budget

Module is 197 lines of Chisel (excluding header comment). The `fu_sketch.md`
estimate for all three conversion lanes was 250 LOC total in
`VFConvLane.scala`; the per-lane size of ~200 lines for one lane suggests the
final number will land around ~500–600 LOC for all three lanes (more verbose
because Chisel has more boilerplate per stage than the pseudocode in the
sketch). Still well within the 1500–3000 LOC overall FU budget.

Area-wise, `fu_sketch.md` estimated the NVFP4→BF16 dequant lane at ~8K NAND2
(≈0.020 mm² @ 16nm). The actual gate count won't be known until Yosys-on-
generic-flow synthesis (Mo 4 work), but the per-lane structure is dominated
by:

- 8x8 unsigned multiplier (~280 NAND2)
- 16-entry × 16-bit LUT (~150 NAND2 combinational; ROM if synthesised as a
  hard macro could be smaller)
- E4M3 subnormal priority encoder + mantissa shuffle (~50 NAND2)
- Three pipeline-register banks (16+16+~5 = ~37 bits, ~200 NAND2 with retiming)
- Normalize / round / pack (~150 NAND2)

Total per lane: ~830 NAND2; 16 lanes ≈ 13K NAND2, ~0.033 mm² @ 16nm. **Higher
than the sketch's 8K estimate**, but still small (~0.7 % of Saturn baseline).
The headline 3 % FU-vs-Saturn ratio still holds.

## Integration plan with Mo 2 microbench

The Mo 2 microbench currently runs software dequant — see
`microbench-mo2/bench_nvfp4.c` for the kernel. Track F lets us drop in a
hardware-FU-emulated dequant in place of the software LUT-based path. Two
ways to slot it in:

**Option A (Mo 4 work):** Compile a Saturn Chipyard SoC that includes the
real `VFConvNvfp4Bf16Lane` instantiated 16× behind the `vfconv.nvfp4.bf16.v`
custom opcode, then re-run Mo 2 on that. This gives the most realistic cycle
count, but requires Saturn integration + decoder work (~2-3 weeks).

**Option B (interim, Mo 3 follow-up):** Add a gem5 cycle-accurate model of
the FU latency (3 cycles, 1 elt/cycle throughput per lane) on top of the
TimingSimpleCPU baseline. The instruction sequence in `bench_nvfp4.c` would
be rewritten using a hand-coded RVV-custom intrinsic that emulates the
opcode (e.g. as inline assembly with a 3-cycle latency on a fake opcode).
This is ~1 day of effort and gives a realistic upper bound on the cycle
improvement before Saturn integration is done.

Recommend Option B for the next checkpoint, since the real value here is
having a quantitative answer to "does Mo 6 require the FU?". My current
prediction: even on a TimingSimpleCPU host, NVFP4-via-FU will land ≥1.05×
faster than FP16 baseline on cycles (vs the current -12% on SW dequant). On
Saturn (BOOM + Saturn vector unit), where dequant pressure on the scalar
core is the dominant overhead, the cycle win should be at least 1.5×.

## Next-up follow-on work

Three near-term derivative tasks now unblocked:

1. **`vfconv.bf16.fp8.v` lane** (~2 hours). Pattern is similar but simpler:
   no per-block scale, 1-cycle pipeline per the sketch. Validation: BF16 ->
   FP8-E4M3 with RNE rounding + saturation; bit-exact vs FP32-to-FP8 round
   golden.

2. **`vfconv.fp8.bf16.v` lane** (~1.5 hours). Even simpler: FP8 -> BF16,
   no multiply, 1-cycle pipeline. Validation: 256 cases bit-exact.

3. **Mo 2 Option B re-measurement** (~1 day). Modify `bench_nvfp4.c` to call
   a 3-cycle-latency stub instead of the software LUT path, re-run gem5,
   record the new cycle/instruction stats. Provides a quantitative bridge
   from "BW reduction" to "decode speedup."

Both 1 and 2 are needed for Mo 4 (FU area ≤ 10 % Saturn) so the area number
covers the full FU not just `vfexp` + `vfconv.nvfp4.bf16.v`. Recommend
running them next session in series; together with item 3 they comprise the
Mo 3 EOM checkpoint.

## Reproducibility

```bash
cd /home/noah/project/riscv/saturn-fu
PATH=/home/noah/.local/share/sbt/bin:$PATH sbt "testOnly saturnfu.VFConvNvfp4Bf16LaneSpec"
```

Sources:
- `src/main/scala/saturnfu/VFConvNvfp4Bf16Lane.scala` (lane RTL)
- `src/test/scala/saturnfu/VFConvNvfp4Bf16LaneTest.scala` (12 tests incl.
  4096-case exhaustive sweep)

Versions:
- Chisel 6.7.0, ChiselTest 6.0.0, Scala 2.13.16 (Chipyard pin)
- sbt 1.10.6 (no-sudo install at `/home/noah/.local/share/sbt/`)

Run time: ~7 seconds total (3 specs, 22 tests, including the 4096-case
exhaustive sweep).

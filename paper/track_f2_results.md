# Track F2 — VFConvBf16Fp8Lane results (2026-05-17)

## Headline

**Track F2 PASS.** 192 LOC Chisel for `VFConvBf16Fp8Lane`, 2-stage pipeline,
21 ChiselTests pass including **0/65536 mismatches across an exhaustive
BF16-input sweep** against a distance-based RNE golden (banker's tie-break,
NaN-encoding saturation). The lane implements `vfconv.bf16.fp8.v` and is the
third of the four Saturn precision-routing custom instructions to reach an
RTL+validation-complete state (after `vfexp.v` from Track E and
`vfconv.nvfp4.bf16.v` from Track F).

| Metric | Value |
|---|---|
| LOC (Chisel) | 192 |
| LOC (test) | 433 |
| Pipeline depth | 2 stages |
| ChiselTests | 21/21 |
| Exhaustive sweep | 65536 BF16 inputs, **0 mismatches** vs RNE golden |
| Cumulative suite | 43/43 tests pass across all 4 modules |
| Latency vs `fu_sketch.md` spec | matches (2 cycles) |
| Throughput | 1 elt/cycle sustained |
| `@instr` decl | `exo_instr_decls.md §5.2` |

## What this is

Per-lane primitive for `vfconv.bf16.fp8.v`: BF16 → FP8-E4M3 (OCP FN
variant) with RNE rounding, saturation at +/-max (=448), and BF16
NaN/Inf pass-through. Used by the FU for the post-softmax cast from BF16
attention weights to FP8 storage in the Q·K^T → P · V mixed-precision
matmul path.

E4M3 variant is the same OCP FN convention as the E4M3 scale in
`VFConvNvfp4Bf16Lane` (exp=15 & m=7 is the only NaN, max normal =
1.110 × 2^8 = 448), so `vfconv.bf16.fp8.v` and `vfconv.fp8.bf16.v` (Track F3,
upcoming) are exact inverses on round-trippable values.

## Pipeline

```
s1 (combinational):
  decode BF16 -> (sign, bexp, bmant)
  classify: is_zero | is_subn | is_inf | is_nan
  compute signed target = bexp - 120  (SInt(10.W))

s1 -> s2 boundary registers latch:
  sign, bmant, target, is_zero_in, is_inf, is_nan

s2 (combinational):
  regime predicates on target:
    target >= 16        -> overflow (saturate +/-max)
    target in [1, 15]   -> normal output
    target in [-3, 0]   -> subnormal output
    target <= -4        -> underflow (round to +/-0)
  build (em_pre, guard, sticky) per regime  (4-way Mux on target for
    subnormal; fixed bit positions for normal)
  RNE: round_up = guard && (sticky || em_pre[0]); em_full = em_pre + round_up
  cascade = em_full[3]
  normal regime: exp += cascade; check post-cascade overflow
    (exp > 15 OR (exp == 15 && em == 7)) -> saturate to (15, 6)
  subnormal regime: exp = cascade ? 1 : 0 (cascade -> min normal 1.000*2^-6)
  final mux: NaN | sat_max (Inf/overflow) | sat_zero (zero/underflow) |
             subn_packed | normal_packed

s2 -> output register
```

Latency = 2 cycles fill, 1 elt/cycle sustained. Backpressure is the same
skeleton pass-through (`io.in.ready := io.out.ready`) used by the other two
lanes; skid buffer remains Mo 4 work.

## Numerical validation

Distance-based golden, independent of the DUT's bit-level shift logic:

1. Pre-build sorted list of all 127 positive E4M3-FN representables (bytes
   `0x00`–`0x7E`, excluding NaN at `0x7F`).
2. For each BF16 input: special-case Inf (→ saturate), NaN (→ sign-preserving
   NaN), zero / subnormal (→ ±0); otherwise convert to Double and find the two
   nearest E4M3 representables via binary search; distance compare with
   banker's tie-break on even byte LSB.
3. Above-max values saturate to `0x7E`/`0xFE`.

**Result: 0/65536 mismatches.** Coverage includes:

- Normal BF16 → normal E4M3 with all RNE outcomes (round-down, round-up,
  cascade across mantissa overflow).
- Saturation at the NaN-encoding boundary (e.g., BF16 = 480 = `0x43F0`, would
  round to byte `0x7F` = NaN but instead saturates to `0x7E` = 448).
- Subnormal output regime (target ∈ {-3, -2, -1, 0}) with all 4 shift
  positions and RNE.
- Subnormal-to-normal cascade (e.g., BF16 = 15/1024 = `0x3C70`, tie at
  7.5 × 2^-9 → banker's even = 8 × 2^-9 = min normal `0x08`).
- Underflow regime (target ≤ -4) and the half-subnormal-grid boundary
  (BF16 = 2^-10 = `0x3A80` → `0x00` via banker's even).
- BF16 ±Inf → ±max FP8 (`0x7E` / `0xFE`).
- BF16 ±NaN → ±NaN FP8 (`0x7F` / `0xFF`, sign-preserved per OCP-FN
  permissive-NaN convention).
- BF16 ±0 → ±0 FP8 (`0x00` / `0x80`).

## Bug-of-record during bring-up

Two unit-test failures on first run; both were arithmetic errors in **my own
test fixtures**, not the DUT — I wrote `0x3BE0` instead of `0x3C60` for the
BF16 representation of `7 × 2^-9` (i.e., used bexp = 119 instead of 120). The
exhaustive 65536-case sweep passed on the first run, which is what tipped me
off: the DUT was correct and only the hand-derived test inputs were wrong.

Lesson: distance-based goldens beat hand-derived expected outputs for FP
quantisation tests. A wrong input bit-pattern won't trigger an exhaustive
sweep failure but will trigger a unit-test failure with a misleading "DUT
gave wrong answer" message. Going forward: when authoring unit tests for FP
conversion, derive expected outputs from the golden function in code rather
than hand-deriving and asserting both sides separately.

No DUT bug-of-record.

## Area / LOC estimate

192 LOC vs the 250-LOC budget for `VFConvLane.scala` in `fu_sketch.md`. The
2-cycle BF16→FP8 lane is the simplest of the four lanes (no multiply, no
LUT, no exp computation), so this comes in well under budget.

Synthesis target (Mo 4): ~4K NAND2 equivalent at 16nm per `fu_sketch.md`
estimate, ≈0.010 mm². The exhaustive-sweep validation gives high confidence
this estimate is realistic (no surprise expansion from missed corner cases).

## Path to Saturn integration

Module is drop-in compatible with `FunctionalUnitFactory` per memory's
verified Saturn pattern:

```scala
val precisionRoutingFU = FunctionalUnitFactory("PrecisionRoutingFU")
  .withCustomOpcodes(Seq(
    CustomOp("vfconv.nvfp4.bf16.v", funct6 = "111000"),
    CustomOp("vfconv.bf16.fp8.v",   funct6 = "111001"),  // <- F2
    CustomOp("vfconv.fp8.bf16.v",   funct6 = "111010"),
    CustomOp("vfexp.v",             funct6 = "111011"),
  ))
  .withLatency(Map(
    "vfconv.bf16.fp8.v" -> 2,
    ...
  ))
```

Per `exo_instr_decls.md §5.2`, the compiler-side `@instr` declaration is
already drafted; `bf16_to_e4m3_rne` extern routes to a `saturn_custom_asm.h`
macro that emits the appropriate `asm volatile(".insn r ...")` once the
funct6 encoding is finalised in the Saturn decoder.

## Next track recommendation

**Track F3** (`vfconv.fp8.bf16.v`, FP8 → BF16, ~1.5 h estimated) is the
natural follow-up: shares the FP8-E4M3 OCP-FN decoder with this lane, no
multiply, 1-cycle pipeline. Completing F3 closes the conversion-lane trio
needed for the Mo 4 area estimate.

Track H (FU-latency-stubbed Mo 2 re-measurement) is the parallel-path
candidate that bridges Mo 2 BW to Mo 6 cycle-speedup. With F2's measured
2-cycle latency now in hand, Track H's latency-stub parameter is one less
unknown.

## File pointers

- DUT: `saturn-fu/src/main/scala/saturnfu/VFConvBf16Fp8Lane.scala` (192 LOC)
- Tests: `saturn-fu/src/test/scala/saturnfu/VFConvBf16Fp8LaneTest.scala` (433 LOC)
- Run: `cd saturn-fu && PATH=/home/noah/.local/share/sbt/bin:$PATH sbt test`
- Spec: `paper/fu_sketch.md §"BF16→FP8-E4M3 quant lane"` (2-cycle target)
- Compiler-side: `paper/exo_instr_decls.md §5.2` (`@instr` decl)
- Precision config: `paper/precision_config.md` (FP8-E4M3 attn-weights line)

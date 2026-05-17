# saturn-fu

Standalone Chisel project for the Saturn precision-routing FU.

Part of the merged X+Y Y1 plan documented in `../paper/`. Specifically the four new RVV custom instructions designed in `../paper/fu_sketch.md`:

| Instruction | Module | Status |
|---|---|---|
| `vfexp.v` | `VFExpLane.scala` | **skeleton** — interface + special cases + 8-stage timing; polynomial math is a placeholder, Mo 3-4 replaces with real degree-5 minimax |
| `vfconv.nvfp4.bf16.v` | (not yet) | not started — Mo 3 |
| `vfconv.bf16.fp8.v`   | (not yet) | not started — Mo 3 |
| `vfconv.fp8.bf16.v`   | (not yet) | not started — Mo 3 |

## Build + test

```bash
PATH=/home/noah/.local/share/sbt/bin:$PATH
cd /home/noah/project/riscv/saturn-fu
sbt compile        # ~5s after first run
sbt test           # ~5s; 5 tests
```

First run downloads Chisel 6.7.0 + ChiselTest 6.0.0 + Scala 2.13.16 from Maven (~150 MB).

## Integration with Saturn (later)

This project deliberately does NOT depend on Chipyard or Saturn. It uses the same Chisel/ChiselTest versions Chipyard pins (`build.sbt` of chipyard root: chisel 6.7.0, chiseltest 6.0.0, scala 2.13.16) so the modules drop into Saturn's `FunctionalUnitFactory` later without version churn.

Mo 3 integration plan: copy the verified `VFExpLane.scala` and conversion-lane modules into `chipyard/generators/saturn/src/main/scala/exu/fp/` and wire via `FunctionalUnitFactory.withCustomOpcodes(Seq(...))`.

## Files

- `build.sbt` — Chisel 6.7.0 + ChiselTest 6.0.0
- `project/build.properties` — sbt 1.10.6
- `src/main/scala/saturnfu/VFExpLane.scala` — vfexp.v lane skeleton (~150 LOC)
- `src/test/scala/saturnfu/VFExpLaneTest.scala` — 5 tests covering interface, latency, special cases, throughput

## Skeleton limitations to remove in Mo 3-4

1. **Placeholder polynomial.** Stages 3-7 currently use a Schraudolph integer-bit-trick approximation that doesn't produce numerically meaningful exp values (it requires a real FP32 multiply to work). The shape-only test confirms timing + no-NaN; the numerical-accuracy test is deferred. Mo 3-4 replaces with degree-5 minimax over [-ln(2)/2, ln(2)/2] using hardfloat FMAs.
2. **No backpressure.** `io.in.ready := io.out.ready` is the dumb pass-through; if downstream stalls mid-pipeline, in-flight data is lost. Mo 3 adds skid buffer.
3. **Single lane.** The full FU is 16 lanes; this is one. Mo 3 generates VFExpLane×16 from a parameterized wrapper.
4. **No subnormal handling.** Inputs in (-2⁻¹³, 0) currently route to "underflow → 0"; some inputs that should produce subnormal exp(x) outputs near the underflow threshold are also flushed to zero. Real implementation needs proper handling.

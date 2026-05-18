# Y2 FireSim Integration Prep (2026-05-18)

Inventory and gate-list for porting the Y1 Saturn-FU artifact to
FireSim for cycle-accurate FPGA-based validation. Cost model assumes
**advisor-borrow on EC2 F1.4xlarge** (per the project's `feedback-
prefer-zero-cost` memory); no purchase path is in scope here.

## Y1 → Y2 transition rationale

Y1 ran the Saturn FUs against gem5 RiscvO3CPU with abstract-issue
FU-latency wiring (`gem5/cpu/o3/FuncUnitConfig.py`). The cycle deltas
in paper §5.6 (Exo PASS at 1.046×) and §7.5 (hand-coded 26-39 %
FU-speedup) are **gem5-level abstract**: they pin FU latency but do
not model the real µarch's vector lane scheduling, register-file
pressure, or chain-cancellation in the Saturn vector pipeline.

FireSim closes that gap. The Saturn RTL goes onto an FPGA at near-
target frequency, the Saturn FUs we built are wired in as **real
ChiselModules**, and the same FA workload runs in a real-time-faithful
simulation. Y2 outputs:

- **Validation that the gem5 cycle counts hold within ±20 %** at the
  real µarch level. If they hold: paper claims are validated, Y2
  publishable as a methodology paper. If they don't: the *direction*
  of the FU benefit is what matters; magnitude may rescale.
- **Yosys 16-nm area numbers** in the paper become standalone-or-
  shared rather than just static-synth — FireSim sees runtime
  utilization, so we can pick the area framing that matches actual
  duty cycle.
- **MLArchSys'27 second-half submission** material (rebuttal / minor
  revision) if the Y1 paper goes to MICRO/ASPLOS/ISCA major-revision
  cycle.

## Existing artifacts to leverage

| Path                                  | What it is                                                                 | FireSim role |
|---------------------------------------|----------------------------------------------------------------------------|--------------|
| `saturn-fu/`                          | Standalone Chisel 6.7.0 project — 5 modules (vfconv-nvfp4↔bf16, vfconv-bf16↔fp8, vfexp) with 57 / 57 ChiselTests passing | Source of FU IP — promote to a Chipyard generator |
| `saturn-fu/build/`                    | Yosys 0.9 synth output, 16 nm liberty map → area numbers in §7.3           | Reference for area-vs-FireSim duty-cycle framing |
| `chipyard/generators/saturn/`         | Berkeley Saturn 25.1 vector unit (in-tree)                                 | Host for the FU plugin point |
| `chipyard/sims/firesim/sim/`          | FireSim 1.20.x simulator harness, target-design framework                  | FireSim platform |
| `microbench-fa/bench_fa_mixed_rvv_native_g14_l{2k,4k,8k,16k}` | Hand-coded FA binaries at 4 seq-lens, GCC 14.2 native FU                | Workloads — re-run on FireSim instead of gem5 |
| `microbench-fa-exo/bench_fa_exo_l{2k,4k,8k,16k}` | Exo-emitted FA binaries (same seq-len sweep)                              | Workloads — Exo PASS verification at FPGA fidelity |
| `paper/figures/fig1_fu_block.tex`     | TikZ FU schematic                                                          | Reference for visual continuity into Y2 paper |

## Y2 task graph (12 weeks, advisor-borrow F1 schedule)

### Phase A — Saturn-FU promotion to Chipyard generator (~2 weeks, cost: $0, local sbt)

1. **Wire `saturn-fu/`'s 5 modules under `chipyard/generators/saturn`'s FU plugin point.**
   - Identify Saturn's FU registration mechanism (likely a Diplomacy
     LazyModule + RoCC-like decoder hook).
   - Refactor saturn-fu/src/main/scala/{...} as a Chipyard sub-
     generator. Keep the 57 / 57 ChiselTests by stubbing the FU host
     interface so unit tests still run standalone.
   - **Risk**: Saturn's FU plug-in API may have changed since the
     decoder patch we did on the gem5 side. Verify against the
     `chipyard/generators/saturn/src/main/scala/` registration API
     before declaring this phase done.
2. **Add a Chipyard config alias** (e.g. `RocketSaturnFAConfig`) that
   instantiates Rocket + Saturn vector unit + our 4 customs.
3. **Smoke test**: `make CONFIG=RocketSaturnFAConfig` under
   `chipyard/sims/verilator/` should produce a Verilator binary that
   runs `bench_fa_mixed_rvv_native_g14_l2k` and matches the gem5
   cycle count within ±10 % (gem5 ran abstract FU latency; Verilator
   runs real RTL — divergence ≤10 % validates the gem5 FU latency
   model was approximately right).

### Phase B — FireSim AGFI build (~1 week, cost: ~$3 / hour × 6 hours = ~$18 F1 time)

1. **Configure** `chipyard/sims/firesim/sim/target-design` with the
   Phase-A config. Pick a single seq-len for the first build (L2K).
2. **Build AGFI**: `make CONFIG=...FA -j$(nproc)`. This is the
   "synthesize + place-and-route" step; runs on F1's host (4 vCPU
   c5 instance is fine) for ~4 h.
3. **Sanity smoke**: AGFI-on-F1 boot test (no workload), confirms
   the design fits on the FPGA.

### Phase C — Workload sweep on FireSim (~1 week, cost: ~$1.65 / hour × 24 hours = ~$40 F1 time)

Run all 8 workloads (hand-coded + Exo, 4 seq-lens each) on FireSim.
Each runs in real time at simulated 3.2 GHz; L2K is ~2 ms, L16K is
~17 ms. With switching/setup overhead the whole sweep is <4 h FPGA
time, but reserve 24 h for re-runs / debug / iteration.

Compare against the gem5 numbers in Table 5 and §5.6. Two outcomes:

- **All cycle deltas within ±20 %**: paper claims validated. Write
  up as "Saturn FAs evaluated end-to-end on FireSim; gem5
  abstract-FU model accurate to ±X %." This is the publishable
  outcome and the main reason to invest the F1 time.
- **>20 % divergence on one or more variants**: investigate. Most
  likely sources: chain-cancellation in Saturn's vector pipeline
  (gem5 doesn't model), register-file write-port contention (gem5
  doesn't model), L1D-miss-then-cache-fill latency under real
  µarch (gem5's `--caches --l1d_size=32KiB` is approximate). For
  each divergent variant, document the µarch effect, and either:
    (a) update the paper's §5/§7 numbers with FireSim as canonical, or
    (b) keep gem5 as the simulator-level claim and add a "FireSim
        validation: ±X % of gem5" caveat in §7.6 or §9.

### Phase D — Llama-3.2-1B end-to-end (~3 weeks, cost: ~$50 F1 time)

Predicated on Phase C succeeding. The single-kernel FA results in
§5/§7 extrapolate to model-level inference latency, but the only way
to *verify* the extrapolation is a real autoregressive decode trace.

1. **Llama-3.2-1B in C** — port from `models/llama.cpp` (in-repo) to
   a static binary that runs under SE mode (no malloc beyond an
   arena, no syscalls beyond exit/m5_dump). Bench harness writes
   one token per gem5_dump tick, compares against fp32 reference.
2. **Run on FireSim**: real-time autoregressive decode, 1024 input
   tokens + 1024 output tokens. Single F1 run is ~15 min wall-clock
   for 1024 output tokens at hardware-real-time.
3. **Headline metric**: tokens/sec at FP32 reference vs Saturn-FU
   mixed-precision. Cross-validate against the §5/§7 per-kernel
   numbers: model-level tokens/sec ≈ kernel-level cycle × tokens
   per kernel × (kernel count per layer × layer count).

### Phase E — Paper assembly (~2 weeks, no F1 cost)

Y2 paper draft. Targets:
- **MICRO 2027** (full paper, June 2027 deadline) if Phase C / D
  succeed and the FireSim data is publishable on its own.
- **MLArchSys'28** (workshop short paper) if the FireSim data is
  validation-only without an independent contribution.

## Total cost model (advisor-borrow EC2 F1)

| Phase | F1 time | EC2 cost (advisor-borrow assumed) |
|-------|---------|-----------------------------------|
| A | 0 | local sbt only |
| B | 4 h F1 + 4 h c5 (host) | ~$18 |
| C | 24 h F1 (with debug margin) | ~$40 |
| D | 50 h F1 + ~$5 storage | ~$55 |
| E | 0 | local LaTeX |
| **Total** | ~78 h F1 | **~$113** |

Advisor-borrow assumption: project doesn't pay for the F1 time. If
advisor borrow falls through, the alternative path is "skip FireSim,
do FPGA-emulation in Verilator at slow speed" (Phase A still gets
Verilator validation; Phases B/C/D get scoped down to Verilator-only
single-kernel runs). Verilator at 100 kHz simulated takes ~10 h per
L16K hand-coded FA workload — feasible but slow. Y2 paper would then
be Verilator-not-FireSim, weaker but free.

## Gate decisions (decision points before committing F1 time)

1. **After Phase A.3 smoke**: if Verilator cycle delta >10 % from
   gem5, the Saturn-FU integration may have a Diplomacy / Decoder
   bug. Fix before paying for F1 time.
2. **After Phase B.3 boot test**: if the design doesn't fit on F1's
   FPGA at 75 MHz host clock, pick a smaller Saturn config (single-
   lane). Re-do B at smaller size.
3. **After Phase C all-runs**: if cycle deltas are within ±5 % of
   gem5 across the board, this is a *validation-only* result and may
   not be a standalone publication (Phase D becomes the publication
   path). If deltas are 5-20 % with clear µarch-attributable patterns,
   Phase C is the publication path on its own (FireSim-corrects-
   gem5 framing).

## What this means for the Y1 paper submission

The Y1 paper is publishable on its own at MLArchSys'27 — gem5 is a
credible methodology for the FU-design-space contribution. The Y2
FireSim work is a follow-up that *strengthens* the Y1 numbers, not a
prerequisite for them. Three frames in §9 ("Future Work"):

a. **Hardware fidelity uplift**: FireSim closes the gem5
   abstract-FU gap; explicitly future-work.
b. **Model-level extrapolation**: Llama-3.2-1B end-to-end ties
   per-kernel cycle to user-visible tokens/sec.
c. **HBM model in gem5**: the §7.6 literal-1.5× threshold is missed
   by 6 % at the DDR3-1600 config; HBM would close that gap
   regardless of FireSim. Independent of A/B but easy to lump in.

The current §9 covers (a) and (b) but is light on (c) — Phase F at
the end of Y2 is a 1-week side-task to add an HBM-1.0 latency model
to gem5 and re-run §7's Table 5 sweep. ~$0 cost (no F1).

## Next concrete action (Y2 kickoff)

1. **Read** `chipyard/generators/saturn/README.md` and the Diplomacy
   FU registration pattern under `chipyard/generators/saturn/src/`.
2. **Stub a saturn-fu/ → chipyard plugin** that imports the 5
   modules without breaking the standalone ChiselTests. Keep
   `paper/track_f_results.md`'s 57 / 57 testbench intact.
3. **Verilator smoke** at L2K to validate the FU integration before
   any F1 spend.

Target start: **Q3 2026** (post-PhD-app cycle). Y1 paper camera-ready
should land before MICRO 2027's June deadline so Phase A's reuse of
artifacts is on stable ground.

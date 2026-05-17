# Kick-off: Y1 RISC-V mixed-prec FA — post-J-4.5f handoff

You are picking up the Y1 RISC-V mixed-precision flash-attention project
on a session boundary. The previous session committed everything to git
(commit `f99fb3d` is the prior baseline; this handoff session adds a
second commit on top). Y1 work is now in a **submission-quality
working-draft state** with all experimental dimensions explored.

## Where the project sits

**Y1 = mixed-prec fused FA on Saturn + NVFP4 K/V + Exo + llama.cpp**,
targeting MLArchSys 2027 workshop (Apr 2027 submission). Multi-paper
trajectory (Y2 HPCA/MICRO 2028, Y3 ISCA/ASPLOS 2029) per
`memory/project_side_project_direction.md` §"Multi-paper trajectory" —
plan is locked, replan Y2-Y4 annually.

### Mo-checkpoint state (cumulative)

| Mo  | Question                                         | Status |
|-----|--------------------------------------------------|--------|
|  2  | NVFP4 K/V ≥30% DRAM BW reduction?                | ✅ PASS — 3.56× verified, holds across L2K-L16K |
|  3  | Custom-instruction RTL bit-exact?                | ✅ PASS — 57/57 ChiselTests, exhaustive sweeps up to 65 K cases/lane |
|  4  | FU area ≤10% of Saturn @ 16 nm est?              | ✅ PASS — 6.34% standalone / 1.10% FMA-shared (Yosys 0.9 → 16 nm) |
|  5  | First FA kernel end-to-end?                       | ✅ PASS — J3+stub fresh O3 = 7.65M cyc |
|  6  | Hand-coded mixed-prec ≥1.5× SpacemiT K1?          | ❌ MISSES across hd∈{64,128} × L2K-L16K, but **gap nearly halved on GCC 14.2 toolchain**: best case L2K hd=64 native/BF16 = **1.59×** (within 6% of literal target). FU integration verified at **26-39 % over SW dequant** at every seq_len. Long-context (L16K) doesn't bridge; hd=128 widens to 1.96×. Paper framing locked as "verified BW + verified FU cycle delta + projected literal speedup conditional on memory subsystem". |
|  8  | Exo-generated kernel within 10% of hand-coded?    | (Track D-follow design DONE; scheduling pass implementation = Y2) |
| 10  | E2E Llama-3.2-1B ≥1.5× K1 iso-power?              | (Y2, depends on FireSim + HBM model in gem5) |

### Artifact state (verified working)

| Artifact | Path | State |
|---|---|---|
| Saturn RTL (5 modules + 57 ChiselTests) | `./saturn-fu/` | sbt test passes |
| gem5 fork (decoder + FU latency + microbenches) | `$GEM5_DIR/` `stable` branch | commit `841d376` |
| Exo fork (SaturnRVV platform + 4 @instrs) | `./exo/` `main` branch | platform file added in this session; smoke test passes |
| Microbenches (kernel + L-sweep + hd-sweep) | `./microbench-fa/` | GCC 14.2 toolchain default |
| Paper draft | `./paper/paper_draft.md` | 9135 words, 9/9 sections drafted, 3 figs, 6 tables |
| Bug report | `./paper/gcc_bug_report.md` | Ready to submit (GCC 13 backport request) |

### Toolchain (load-bearing)

- **GCC 14.2** at `/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/`
- Flags: `-O2 -fno-tree-vectorize` (workaround for GCC 14 -O2 auto-vec bug)
- Pre-widen-Q workaround in `bench_fa_mixed_rvv_native.c` retained (defensive)
- GCC 13.2 at `/path/to/bootlin-riscv64/bin/` retained as fallback (older measurements use this)
- gem5 conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build`

## Open items (none gate paper submission)

### Highest-leverage in-session work

1. **Verify open-source artifact public-readiness** (~30 min). Walk
   through `saturn-fu/`, `gem5/`, `exo/`, `microbench-fa/`: confirm
   no credentials, README pointers to `paper/` reproduce-recipes,
   git states clean. The paper's headline open-source claim is
   load-bearing — if camera-ready ships before repos go public,
   the contribution is hollow. Currently no repos are pushed to
   any public host.
2. **Camera-ready polish** (~1 h). Citation standardization;
   convert ASCII figures to SVG/TikZ; abstract → 200-word trim if
   venue requires; compress the 5× repetition of "Mo 6 misses"
   message; verify Table 1 (intro_draft.txt → paper_draft.md
   Table 6) is fully refreshed.
3. **Submit GCC bug reports** (~15 min, requires bugzilla account).
   `gcc_bug_report.md` is ready. One is a GCC 13 backport request
   (well-characterized fix already exists in 14.2+); the second
   GCC 14 -O2 auto-vec issue is mentioned but not minimized — would
   need a minimal reproducer first if filing.

### Y2 work (substantial, multi-session)

4. **FireSim integration + HBM bandwidth model in gem5**. The
   "literal ≥1.5× projected on HBM-class memory" claim needs
   verification before any tier-1 conference submission. Either
   path closes the projected-vs-verified gap; FireSim also unlocks
   real-Saturn-µarch validation (the gem5 single-instruction-3c-FU
   stub underestimates 16-lane FU benefit by ~3-4×).
5. **E2E Llama-3.2-1B on Saturn-FU** (Mo 10). Needs FireSim + a
   K1 baseline (advisor borrow abandoned; published numbers or
   FireSim emulation only).
6. **Exo scheduling pass implementation**. The `@instr` library
   is implemented + smoke-tested; the pass that takes a high-level
   `@proc` and emits the §6 kernel via `replace()` calls is the
   Mo 8 deliverable.
7. **ARCQuant perplexity sweep for §3** (gates §3 from placeholder
   to drafted prose). Needs GPU + ARCQuant fork + Llama-3.2-1B.
8. **NVFP4 calibration on Llama-3.2-1B**. Same setup as #7.

### Strategic (not in-session, but the calendar)

9. **Advisor pitch**. Per `user_phd_application.md`, the advisor
   has not yet been pitched on the side project as of 2026-05-16.
   `paper/advisor_msg.txt` was drafted but is now stale (Path A
   silicon borrow abandoned). A fresh pitch leading with the Y1
   results (MLArchSys 2027 workshop submission, open-source
   artifact, FireSim Y2 plan) would be high-value.

## Memory + git context

- Project repo at `./.git` on `main`, two
  commits (initial `f99fb3d` + this session's handoff commit).
- Exo fork at `./exo/.git` on `main`, one
  commit on top of upstream `2f5472d` (the SaturnRVV platform).
- gem5 fork at `$GEM5_DIR/.git` on `stable`,
  commit `841d376` (decoder + FU latency wiring; not modified this
  session).
- saturn-fu lives inside the main project repo (no separate .git);
  RTL committed in the prior session's `f99fb3d`.
- Auto-memory current at `~/.claude/projects/-home-noah-project-riscv/memory/`.

## Operating mode (preserved from prior memory)

- `feedback-direct-recommendations`: give the technical playbook
  with reasoning, not scope-narrowing.
- `feedback-keep-work-dir-clean`: ephemeral scratch → /tmp, durable
  artifacts → `./paper/`.
- `feedback-rvv-asm-vsetvli`: disassemble + check vsetvli stream
  before believing operand-width hypotheses (GCC 13.2 pass bug);
  GCC 14.2+ has the fix but introduced a new -O2 auto-vec issue
  requiring `-fno-tree-vectorize`.
- `feedback-fu-stub-brackets`: when projecting a HW FU on a
  non-HW simulator, bracket conservative + aggressive stubs.
- `feedback-distance-based-goldens`: for FP-conversion Chisel
  tests, build the golden as distance search + exhaustive sweep.

## Suggested first move

Pick from §"Highest-leverage in-session work" above. **Option 1
(open-source artifact verification)** has the highest direct impact
on the paper-headline claim. **Option 3 (GCC bug submission)** is
the smallest, but requires a bugzilla account so is user-action,
not assistant-executable. **Option 2 (camera-ready polish)** is
mechanical but real-value work.

If the next session is strategic rather than execution-focused
(e.g., write the advisor pitch, plan Y2 in detail), say so and the
work will redirect accordingly.

Go.

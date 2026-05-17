# Kick-off: Y1 RISC-V mixed-prec FA — post-camera-ready-polish handoff

You are picking up the Y1 RISC-V mixed-precision flash-attention
project on a session boundary. The previous session committed three
follow-ups on top of `f99fb3d` (initial) / `1e76899` (J-sweep+paper):
artifact public-readiness, polish follow-up, and camera-ready polish.

```
87f7743  Y1 camera-ready polish: abstract trim, Mo 6 compression, citation cleanup
db561e0  Y1 artifact polish follow-up: per-bench READMEs + sanitize /home/noah
fb73ef7  Y1 artifact polish: README, LICENSE, gitignore, de-hardcode microbench-j4
1e76899  Y1 J-4.5b through J-4.5f: complete experimental sweep + paper draft
f99fb3d  Initial commit: Y1 RISC-V mixed-prec FA project sources + writeups
```

Y1 is in a **submission-quality, public-pushable working-draft state**
with all experimental dimensions explored, the open-source artifact
audited for public release, and the paper draft trimmed to workshop
conventions.

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
|  6  | Hand-coded mixed-prec ≥1.5× SpacemiT K1?          | ❌ MISSES across hd∈{64,128} × L2K-L16K, **best L2K hd=64 native/BF16 = 1.59× (within 6% of literal target)**. FU integration verified at 26-39% over SW dequant at every seq_len. Paper framing: verified BW (3.56×) + verified FU cycle delta (26-39%) + literal-≥1.5× projected on HBM-class memory. |
|  8  | Exo-generated kernel within 10% of hand-coded?    | (Track D-follow design DONE; scheduling pass implementation = Y2) |
| 10  | E2E Llama-3.2-1B ≥1.5× K1 iso-power?              | (Y2, depends on FireSim + HBM model in gem5) |

### Paper draft state (post camera-ready polish)

- **9/9 sections drafted.** §3 (Per-Stage Precision Analysis) is
  still OUTLINE-state gated on the ARCQuant perplexity sweep —
  every other section is in working-draft prose.
- **Abstract: 246 words** (workshop-convention 200-250).
- **Mo 6 caveat** is consolidated in §7.6; §1 + §9 are glance refs.
- **Citations standardized**: 5 bare-`[arXiv:NNNN]` brackets
  converted to `[Anonymous, arXiv'26]` (ARCQuant) or
  `[arXiv:NNNN, YYYY]` (SystolicAttention, MR-GPTQ, IREE-RVV).
- **§6.1 pseudocode** now carries an FU-lane mapping footnote.
- **Table 6 reconciled with Table 1** (FuseMax OSS column → "no"
  per the model-only-Timeloop risk flag).
- 3 ASCII figures still inline; **ASCII → SVG/TikZ deferred**
  (separate multi-hour job; ASCII renders adequately).

### Artifact state (verified public-ready)

| Artifact | Path | State |
|---|---|---|
| Saturn RTL (5 modules + 57 ChiselTests + synth scripts) | `./saturn-fu/` | sbt test passes; current README; LICENSE in repo root |
| gem5 fork (decoder + FU latency + microbenches) | `./gem5/` `stable` branch | commit `841d376`; clean working tree |
| Exo fork (SaturnRVV platform + 4 @instrs) | `./exo/` `main` branch | commit `15d61db`; smoke test passes |
| Microbenches (kernel + L-sweep + hd-sweep) | `./microbench-{mo2,j4,fa}/` | GCC 14.2 default; per-bench READMEs; Makefiles for all three |
| Paper draft | `./paper/paper_draft.md` | 246-word abstract, 9/9 sections drafted, 3 figs, 6 tables |
| Bug report | `./paper/gcc_bug_report.md` | Ready to submit (GCC 13 backport request) |
| Top-level orientation | `./README.md`, `./LICENSE` | BSD-3-Clause, paper headline + reproduce-recipe map |

**Public-readiness verified**: no credentials, no `/home/noah` paths
in source (15 paper docs sanitized to `./`-relative + `$GEM5_DIR` env
var), READMEs current, LICENSE in place. No repos pushed to any
public host yet — release-step pending user action.

### Toolchain (load-bearing)

- **GCC 14.2** at `/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/`
- Flags: `-O2 -fno-tree-vectorize` (workaround for GCC 14 -O2 auto-vec bug)
- Pre-widen-Q workaround in `bench_fa_mixed_rvv_native.c` retained (defensive)
- GCC 13.2 retained as fallback (older measurements use this)
- gem5 conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build`

## Open items (none gate paper submission)

### Highest-leverage in-session work

1. ~~Verify open-source artifact public-readiness~~ ✅ **DONE**
   (`fb73ef7` + `db561e0`). 5 blockers fixed (top-level README +
   LICENSE, saturn-fu gitignore + README, microbench-j4 hardcoded
   paths); 4 polish items landed (per-bench READMEs, Makefile path
   cleanup, microbench-j4 Makefile, paper-docs sanitization).
2. ~~Camera-ready polish~~ ✅ **DONE** (`87f7743`). Abstract trim
   (307 → 246), Mo 6 compression, citation standardization,
   §6.1 pseudocode footnote, Table 6 reconciliation. ASCII →
   SVG/TikZ figures deferred.
3. **Submit GCC bug reports** (~15 min, requires bugzilla account).
   `gcc_bug_report.md` is ready. One is a GCC 13 backport request
   (well-characterized fix already exists in 14.2+); the second
   GCC 14 -O2 auto-vec issue is mentioned but not minimized — would
   need a minimal reproducer first if filing.
4. **Push public repos**. Saturn fork (part of main repo), gem5
   fork, Exo fork, and main project repo are all artifact-clean
   but unpushed. Camera-ready paper headline claim ("All RTL,
   gem5 patches, kernels, and Exo declarations open-source") is
   load-bearing — must precede paper submission. The LICENSE
   copyright line is a placeholder ("The Precision-Routing RVV
   Flash Attention Project Authors") that should be edited before
   any public push.

### Y2 work (substantial, multi-session)

5. **FireSim integration + HBM bandwidth model in gem5**. The
   "literal ≥1.5× projected on HBM-class memory" claim needs
   verification before any tier-1 conference submission. Either
   path closes the projected-vs-verified gap; FireSim also unlocks
   real-Saturn-µarch validation (the gem5 single-instruction-3c-FU
   stub underestimates 16-lane FU benefit by ~3-4×).
6. **E2E Llama-3.2-1B on Saturn-FU** (Mo 10). Needs FireSim + a
   K1 baseline (advisor borrow abandoned; published numbers or
   FireSim emulation only).
7. **Exo scheduling pass implementation**. The `@instr` library
   is implemented + smoke-tested; the pass that takes a high-level
   `@proc` and emits the §6 kernel via `replace()` calls is the
   Mo 8 deliverable.
8. **ARCQuant perplexity sweep for §3** (gates §3 from placeholder
   to drafted prose). Needs GPU + ARCQuant fork + Llama-3.2-1B.
9. **NVFP4 calibration on Llama-3.2-1B**. Same setup as #8.
10. **ASCII figures → SVG/TikZ** (~3-4 h). §4 FU block diagram,
    §6 kernel dataflow, §7 cycle-stack bar chart. Defer to
    camera-ready revision.

### Strategic (not in-session, but the calendar)

11. **Advisor pitch**. Per `user_phd_application.md`, the advisor
    has not yet been pitched on the side project as of 2026-05-16.
    `paper/advisor_msg.txt` was drafted but is now stale (Path A
    silicon borrow abandoned). A fresh pitch leading with the Y1
    results (MLArchSys 2027 workshop submission, open-source
    artifact, FireSim Y2 plan) would be high-value.

## Memory + git context

- Project repo at `./.git` on `main`, **five commits**
  (`f99fb3d` → `1e76899` → `fb73ef7` → `db561e0` → `87f7743`).
- Exo fork at `./exo/.git` on `main`, one commit on top of upstream
  `2f5472d` (the SaturnRVV platform).
- gem5 fork at `./gem5/.git` on `stable`, commit `841d376`
  (decoder + FU latency wiring; not modified this session).
- saturn-fu lives inside the main project repo (no separate `.git`);
  RTL committed in `f99fb3d`, synth scripts added in `fb73ef7`.
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

The natural ordering on what's left:

- **Option 4 (push public repos)** has the highest direct impact on
  the paper-headline claim — without it the "open-source" claim is
  hollow. Requires the user to edit the LICENSE copyright placeholder
  and run `git push` against host(s) of their choice. Not
  assistant-executable without user action.
- **Option 3 (submit GCC bug reports)** is small and bounded but
  requires a bugzilla account, so also user-action.
- **Option 11 (advisor pitch)** is the highest-leverage strategic
  next step. Rewriting `paper/advisor_msg.txt` from scratch around
  the Y1 results (workshop submission + open-source artifact +
  FireSim Y2 plan) is assistant-executable and the right framing
  for a real conversation with the advisor.
- **Option 7 (Exo scheduling pass)** is the highest-impact technical
  next step — closes Mo 8, completes the compiler-co-design story.
  Multi-session.

If the next session is strategic (option 11) or release-prep
(options 3, 4), say so. Otherwise the technical path is option 7
(Mo 8) or option 8/9 (Mo §3-perplexity-sweep).

Go.

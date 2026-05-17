# Kick-off: Y1 RISC-V mixed-prec FA — post-Mo-8-foundation handoff

You are picking up the Y1 RISC-V mixed-precision flash-attention
project on a session boundary. The previous session landed the **Mo 8
foundation** on top of the artifact-readiness + camera-ready-polish
arc:

Main repo (`./.git` on `main`):
```
694611e  Mo 8 foundation: Exo scheduling pass demo (dequant + softmax-exp chunks)
039692e  Y1 handoff kickoff: refresh for post-camera-ready-polish session boundary
87f7743  Y1 camera-ready polish: abstract trim, Mo 6 compression, citation cleanup
db561e0  Y1 artifact polish follow-up: per-bench READMEs + sanitize /home/noah
fb73ef7  Y1 artifact polish: README, LICENSE, gitignore, de-hardcode microbench-j4
1e76899  Y1 J-4.5b through J-4.5f: complete experimental sweep + paper draft
f99fb3d  Initial commit: Y1 RISC-V mixed-prec FA project sources + writeups
```

Exo fork (`./exo/.git` on `main`):
```
d21ad05  Saturn RVV platform: add vle/vse @instrs + fix vfexp_v body for unification
15d61db  Add Saturn RVV platform: 4 custom @instr decls + memory class
2f5472d  Bump dependencies (#820)
```

gem5 fork (`./gem5/.git` on `stable`): unchanged, `841d376`.

Y1 is in a **submission-quality, public-pushable, compiler-foundation-
landed working-draft state**.

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
|  8  | Exo-generated kernel within 10% of hand-coded?    | 🟡 **FOUNDATION LANDED 2026-05-17 LATE-LATE.** `paper/exo_schedule_fa.py` demonstrates end-to-end scheduling pass on two chunks (NVFP4 dequant + vfexp softmax); emitted C cross-compiles cleanly + disassembly contains Saturn `.4byte` encodings. Mo 8 endpoint (cycle parity with hand-coded native, ≥0.9×) is multi-session follow-on. |
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
- **§5.5 framing** currently calls the precision-routing pass a
  "Y2 deliverable" — outdated post Mo 8 foundation; should be
  rewritten in a future polish pass to say the foundation has
  landed and the scaling-to-full-FA is the Mo 8 follow-on.
- 3 ASCII figures still inline; **ASCII → SVG/TikZ deferred**.

### Artifact state (verified public-ready + Mo 8 foundation)

| Artifact | Path | State |
|---|---|---|
| Saturn RTL (5 modules + 57 ChiselTests + synth scripts) | `./saturn-fu/` | sbt test passes; current README; LICENSE in repo root |
| gem5 fork (decoder + FU latency + microbenches) | `./gem5/` `stable` branch | commit `841d376`; clean working tree |
| Exo fork (SaturnRVV platform + 4 customs + 6 vle/vse @instrs) | `./exo/` `main` branch | commit `d21ad05`; smoke test passes; **scheduling pass demos compile + cross-compile** |
| Microbenches (kernel + L-sweep + hd-sweep) | `./microbench-{mo2,j4,fa}/` | GCC 14.2 default; per-bench READMEs; Makefiles for all three |
| **Exo scheduling demo** | `./paper/exo_schedule_fa.py` | **NEW**; 215 LOC; two end-to-end demos lowering @procs onto SaturnRVV |
| Paper draft | `./paper/paper_draft.md` | 246-word abstract, 9/9 sections drafted, 3 figs, 6 tables |
| Bug report | `./paper/gcc_bug_report.md` | Ready to submit (GCC 13 backport request) |
| Top-level orientation | `./README.md`, `./LICENSE` | BSD-3-Clause, paper headline + reproduce-recipe map |

**Public-readiness verified**: no credentials, no `/home/noah` paths
in source, READMEs current, LICENSE in place. **No repos pushed
to any public host yet** — release-step pending user action.

### Toolchain (load-bearing)

- **GCC 14.2** at `/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/`
- Flags: `-O2 -fno-tree-vectorize` (workaround for GCC 14 -O2 auto-vec bug)
- Pre-widen-Q workaround in `bench_fa_mixed_rvv_native.c` retained (defensive)
- GCC 13.2 retained as fallback (older measurements use this)
- gem5 conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build`
- Exo install: `pip install -e exo/` (editable install used by `paper/exo_schedule_fa.py`)

## Open items (none gate paper submission)

### Highest-leverage in-session work

1. ~~Verify open-source artifact public-readiness~~ ✅ **DONE**
   (`fb73ef7` + `db561e0`).
2. ~~Camera-ready polish~~ ✅ **DONE** (`87f7743`).
3. **Submit GCC bug reports** (~15 min, requires bugzilla account).
   `gcc_bug_report.md` is ready — GCC 13 backport request.
4. **Push public repos**. The "open-source" claim in the paper
   abstract is load-bearing. LICENSE copyright line is a placeholder
   ("The Precision-Routing RVV Flash Attention Project Authors")
   that should be edited before any public push.
5. ~~Exo scheduling pass FOUNDATION~~ ✅ **DONE** (`d21ad05` in
   exo fork + `694611e` in main).

### Y2 work (substantial, multi-session)

6. **FireSim integration + HBM bandwidth model in gem5**. The
   "literal ≥1.5× projected on HBM-class memory" claim needs
   verification before any tier-1 conference submission.
7. **E2E Llama-3.2-1B on Saturn-FU** (Mo 10). Needs FireSim + a
   K1 baseline (published numbers or FireSim emulation only).
8. **Mo 8 cycle parity follow-on** (continues the foundation):
   - Tile the 16-lane @proc to head_dim=64 chunks via `divide_loop`.
   - Compose §6 FA structure: 8 heads × seq_len × QK^T + softmax
     max/sum reductions + P-quant + P·V passes.
   - Wire the remaining 2 vfconv lanes (`bf16.fp8.v` for FP8 quant,
     `fp8.bf16.v` for P·V dequant) into the schedule.
   - Build the Exo-generated kernel on gem5 + compare cycles to
     `bench_fa_mixed_rvv_native`. Target: within 10% (Mo 8 PASS).
9. **ARCQuant perplexity sweep for §3** (gates §3 from placeholder
   to drafted prose). Needs GPU + ARCQuant fork + Llama-3.2-1B.
10. **NVFP4 calibration on Llama-3.2-1B**. Same setup as #9.
11. **ASCII figures → SVG/TikZ** (~3-4 h). §4 FU block diagram, §6
    kernel dataflow, §7 cycle-stack bar chart. Defer to
    camera-ready revision.
12. **Paper §5.5 polish**: rewrite to note the precision-routing
    pass foundation has landed; reframe Mo 8 follow-on as "scaling
    the demonstrated foundation to the full §6 kernel shape."

### Strategic (not in-session, but the calendar)

13. **Advisor pitch**. Per `user_phd_application.md`, the advisor
    has not yet been pitched on the side project as of 2026-05-16.
    A fresh pitch leading with the Y1 results (workshop submission +
    open-source artifact + Mo 8 foundation + FireSim Y2 plan) would
    be high-value.

## Memory + git context

- Main project repo at `./.git` on `main`, **seven commits**
  (`f99fb3d` → `1e76899` → `fb73ef7` → `db561e0` → `87f7743` →
  `039692e` → `694611e`).
- Exo fork at `./exo/.git` on `main`, **two commits** on top of
  upstream `2f5472d` (the SaturnRVV platform `15d61db` + the
  vle/vse + vfexp-body fix `d21ad05`).
- gem5 fork at `./gem5/.git` on `stable`, commit `841d376`
  (decoder + FU latency wiring; not modified this session).
- saturn-fu lives inside the main project repo (no separate `.git`).
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

## Notes for the Mo 8 follow-on

The scheduling pass demo (`paper/exo_schedule_fa.py`) hit two
non-obvious Exo idioms worth recording for the scaling sessions:

1. **`replace()` matches loops by iter-var pattern**. After
   `stage_mem`, new load/store loops auto-rename their iter var to
   `i0` (to avoid colliding with the original `i`). When chaining
   multiple `replace()` calls, the pattern string must match the
   actual iter name post-stage_mem, not the original `i`.
2. **Unification refuses unused @instr parameters.** An @instr
   whose opaque body doesn't reference all its declared parameters
   will fail unification with `UnificationError: un-unused argument`.
   `vfexp_v`'s body was originally `dst[i] = 0.0` (didn't reference
   `src`); this session changed it to `dst[i] = src[i]` even
   though the types mismatch — same opaque/type-mismatch idiom as
   `vfconv.bf16.fp8` (ui8 ← ui16).

## Suggested first move

The natural ordering on what's left:

- **Option 4 (push public repos)** has the highest direct impact on
  the paper-headline claim. Requires user to edit the LICENSE
  copyright placeholder and run `git push`. Not assistant-executable
  without user action.
- **Option 3 (submit GCC bug reports)** is small and bounded but
  requires a bugzilla account, so also user-action.
- **Option 8 (Mo 8 cycle parity follow-on)** is the natural
  technical continuation. ~2-3 sessions to reach within-10%-of-hand-
  coded parity. Foundation already proved the methodology works.
- **Option 13 (advisor pitch rewrite)** is the highest-leverage
  strategic next step. Now an even stronger pitch with the Mo 8
  foundation landed.

If the next session is strategic (option 13) or release-prep
(options 3, 4), say so. Otherwise the natural technical path is
option 8 (scale Mo 8 foundation toward cycle parity).

Go.

# Next-session warmup prompt

Copy-paste this into a fresh Claude Code session at `./` to start the
next session on this project. Y1 is COMPLETE (Mo 8 cycle-parity done,
public repos pushed, paper at camera-ready); Y2 + Y3 directions LOCKED
+ DOC'D as of 2026-05-19; **Y2 prep early-pickup landed 2026-05-19
evening** — Llama-3.2-1B-Instruct F16 GGUF on disk, x86 ppl harness
operational, uniform-F16 baseline ppl = 11.9048 ± 0.0843.

---

```
Read paper/y1_handoff_kickoff.md, paper/y2_hetero_precision_routing.md,
paper/y3_spec_decode.md, paper/y2_session1_notes.md, and the cumulative
project memory at ~/.claude/projects/-home-noah-project-riscv/memory/
to load context. Current state: Y1 COMPLETE (3 GitHub repos live at
github.com:noyaboy/{precision-routing-rvv-fa, exo-saturn-rvv,
gem5-saturn-fu}, MLArchSys'27 submission Apr 2027); Y2 LOCKED
(per-layer precision policy → MICRO'28 primary, May-Jul 2027 gate);
Y3 LOCKED (spec decode on Saturn → ISCA'29 primary, Q3/Q4 2027 gate).
Y2 prep partially landed 2026-05-19 evening: Llama-3.2-1B-Instruct F16
GGUF + x86 llama.cpp build (build/llama-x86/) + WikiText-2 raw +
uniform-F16 ppl baseline 11.9048 ± 0.0843 (ctx=2048, 141 chunks,
Instruct variant). Next deliverable per paper/y2_session1_notes.md is
the NVFP4 K/V quant path through llama.cpp's RVV decoder. Then suggest
a starting move based on today's date.
```

---

## What the next session has to work with

- **Main repo** at `./` on `main`, public at
  `git@github.com:noyaboy/precision-routing-rvv-fa.git`.
- **Exo fork** at `./exo/` on `main`, public at
  `git@github.com:noyaboy/exo-saturn-rvv.git`. SaturnRVV memory class
  + 4 Saturn custom `@instr`s + 6 vle/vse @instrs + vfexp body fix.
- **gem5 fork** at `./gem5/` on `stable`, public at
  `git@github.com:noyaboy/gem5-saturn-fu.git`. Saturn FU latency
  config + RISC-V decoder hooks for the 4 customs.
- **Memory** at `~/.claude/projects/-home-noah-project-riscv/memory/`
  with project-state record + 11 feedback memories + 1 user + 1
  reference.
- **Y1 paper** at `paper/paper_draft.md` (camera-ready quality;
  Llama-3.2-1B layer count corrected 16, not 24, on 2026-05-19).
- **Y2 scoping** at `paper/y2_hetero_precision_routing.md` (~17 KB;
  all 7 §9 open questions answered with recommendations).
- **Y3 scoping** at `paper/y3_spec_decode.md` (~20 KB; 11 sections
  symmetric to Y2; landscape corrections vs project memory's earlier
  SPAD-as-spec-decode framing).
- **FireSim validation infra** at `paper/y2_firesim_prep.md`
  (repurposed as Y2 fallback + Y3 validation infra).
- **Y2 prep session 1 state** at `paper/y2_session1_notes.md`
  (2026-05-19 evening): early-pickup work landed —
  `models/Llama-3.2-1B-Instruct-f16.gguf` (SHA `1f33ad43…`),
  `build/llama-x86/` (separate from rv64gcv build,
  AVX-native + OpenMP), `datasets/wikitext-2-raw/`, baseline
  uniform-F16 PPL = **11.9048 ± 0.0843** on WikiText-2 test
  (ctx=2048, 141 chunks, Instruct variant — not base; Δppl
  gate signal unaffected).

## Date-anchored intents (pick based on today's date)

- **May 2026 - Jul 2026 (dormant default; PARTIAL OPT-OUT
  2026-05-19)**: side project default is inactive — main-group
  research is the line. Only proceed if user explicitly opts out
  of dormancy. **Note**: 2026-05-19 evening already executed 4 of
  the Aug-2026 first tasks (GGUF download, x86 ppl harness, base
  dataset, uniform-F16 ppl baseline). Next-session pickup should
  NOT redo these — read `paper/y2_session1_notes.md` first.
- **Aug 2026 - Oct 2026 (Y2 scoping pickup)**: warm-start from
  `paper/y2_hetero_precision_routing.md` §9 (all questions
  pre-answered). First concrete tasks per the doc (✅ marks
  ones already landed 2026-05-19 evening):
  - ✅ Download Llama-3.2-1B GGUF (F16 Instruct, on disk)
  - ✅ Uniform-F16 PPL baseline (11.9048 ± 0.0843; gate iso-acc
    threshold ε = 12.0048 ppl)
  - NVFP4 K/V quant path through llama.cpp's RVV decoder
    (uniform first, then per-layer) — natural next session
  - Run uniform-NVFP4-except-band sweep (6 configs, ~1 hr) per Q2
  - Run K-knee policy sweep (~overnight) per Q5
- **Nov 2026 - Mar 2027 (Y1 polish + Phase A dual-use)**: Y1
  camera-ready writing + arXiv LaTeX conversion. **In parallel**:
  FireSim Phase A (Chipyard integration + Verilator smoke) per
  Q7 — dual-use validation of gem5 FU latency model before
  MLArchSys'27 camera-ready freezes.
- **Apr 2027 (MLArchSys'27 submission)**: file MLArchSys'27 paper.
- **May-Jul 2027 (Y2 gate decision)**: per `y2_hetero_precision_
  routing.md` §6.2 decision tree (≥10% → commit / 5-10% →
  tighten / <5% → FireSim-validation fallback).
- **Aug 2027 - Mar 2028 (Y2 execution if gate-passed)**: FireSim
  Phase B-D (~$113 EC2 F1 advisor-borrow) + MICRO'28 paper drafting.
  Apr 2028 = MICRO'28 submission.
- **Q3/Q4 2027 (Y3 gate decision)**: per `y3_spec_decode.md` §6
  (≥1.5× tok/s vs Y2 baseline → commit / 1.2-1.5× → tighten /
  <1.2× → FireSim-validation fallback).
- **Jan 2028 - Q4 2028 (Y3 execution if gate-passed)**: spec-verify
  FU + KV-rollback HW + Exo batched-verify scheduling. ISCA'29
  submission ~Nov 2028.
- **2029 (polish)**: arXiv revisions, application materials.
- **Dec 2029 / Jan 2030**: PhD application deadlines.

## Deferred user-actions (not assistant-executable)

- **GCC Bugzilla submission** — prep at `paper/gcc_bugzilla_
  submission.md` + verified reproducers at `paper/gcc_repro/`. Must
  be filed via Bugzilla web UI per `feedback-prefer-web-git-over-gh`.
- **LICENSE attribution update** — post-blind-review; current
  BSD-3-Clause stub awaits MLArchSys'27 accept/reject.
- **Advisor pitch** — DEFERRED per `feedback-defer-advisor-pitch`.
  Side project is for PhD admissions + MLArchSys'27, NOT current
  advisor. Do NOT suggest pitching as next step.

## Operating mode (durable; all in `~/.claude/projects/.../memory/`)

- `user_phd_application`: applying to US Architecture PhDs (Fall
  2030 entry, Dec 2029 / Jan 2030 deadlines); LLM accel focus;
  full-stack is a hard requirement; main-group research separate
  from side project.
- `feedback_defer_advisor_pitch`: do NOT recommend pitching side
  project to current advisor.
- `feedback_direct_recommendations`: give the technical playbook
  with reasoning, NOT scope-narrowing or career redirection.
- `feedback_keep_work_dir_clean`: ephemeral scratch → /tmp/,
  durable artifacts → `/home/noah/project/riscv/paper/`.
- `feedback_prefer_zero_cost`: default to advisor-borrow + free
  simulation; surface tradeoffs, don't push purchases as hidden
  assumptions.
- `feedback_prefer_web_git_over_gh`: web UI + plain `git`, NOT
  `gh` CLI, for repo operations.
- `feedback_shallow_fork_push`: check
  `git rev-parse --is-shallow-repository` before `git push` of a
  fork; `git fetch --unshallow upstream` if shallow.
- `feedback_recheck_upstream_state`: re-grep current code before
  committing to "fills the gap in X" framing.
- `feedback_rvv_asm_vsetvli`: GCC 13.2's vsetvli pass doesn't see
  across inline asm; disassemble first when RVV numerics break.
- `feedback_distance_based_goldens`: for FP-conversion Chisel
  tests, build the golden as distance search + exhaustive sweep.
- `feedback_fu_stub_brackets`: bracket FU projections with
  conservative + aggressive stubs.
- `feedback_exo_scheduling_idioms`: 6 Exo + Saturn-platform
  gotchas (replace iter-name, unification, stage_mem global pool,
  simplify before set_memory, FA-shape vs methodology demo,
  intrinsic-vs-asm-volatile ~12× cost).

## Watch-list (re-grep monthly during active windows; per
`feedback_recheck_upstream_state`)

- **Benini group** (post-VMXDOTP DATE'26) — RVV-customs-for-ML
  hazard. Wipfli moved to Axelera AI; group still active on Spatz.
- **Han lab** — LServe (MLSys'25, arXiv:2502.14866) uses "unified
  sparse attention framework" framing uncomfortably close to Y2's
  "compiler-scheduled per-layer policy." Sustained sparse-attention
  + serving activity. Y2/Y3 adjacency risk persists.
- **Ragan-Kelley group** — Exo trunk `src/exo/platforms/rvv.py`
  still f32+m1-only stub (verified 2026-05-19); no RVV scoop
  activity. Our `exo-saturn-rvv` fork's delta is wide open.
- **Hansung Kim** (Berkeley `gemmini-mx`) — no new gemmini-mx
  scoop activity surfaced (May 2026); send technical-courtesy
  email per Y2 doc §9 Q6.
- **Spec-decode HW** — HADES (arXiv:2412.19925, Dec 2024) is real
  Y3 precedent (NOT SPAD as earlier memory claimed); SPEQ +
  SpecMamba + From-Quarter-to-All + Traversal Verification all
  exist; no public RVV/Saturn spec-decode work yet.

## Quick state-check commands

```bash
# Confirm Y1 public artifact integrity (no rebase / push needed)
git -C ./ log --oneline -1
git -C ./exo log --oneline -1
git -C ./gem5 log --oneline -1

# Confirm Y2 + Y3 docs present
ls -la paper/y2_hetero_precision_routing.md paper/y3_spec_decode.md \
       paper/y2_session1_notes.md

# Confirm Llama-3.2-1B downloaded + verify SHA (2026-05-19 evening)
ls -la models/Llama-3.2-1B-Instruct-f16.gguf  # expect 2,479,595,360 B
sha256sum models/Llama-3.2-1B-Instruct-f16.gguf  # expect 1f33ad43…

# Confirm x86 ppl harness ready (re-run only if invalidating baseline)
ls -la build/llama-x86/bin/llama-perplexity datasets/wikitext-2-raw/
# Baseline PPL is 11.9048 ± 0.0843 (recorded in y2_session1_notes.md)
```

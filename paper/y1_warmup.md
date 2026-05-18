# Y1 next-session warmup prompt

Copy-paste this into a fresh Claude Code session at
`./` to start the next session on this project.

---

```
Read paper/y1_handoff_kickoff.md and the cumulative project memory at
~/.claude/projects/-home-noah-project-riscv/memory/ to load context.
The current state is post-Mo-8-foundation: paper/exo_schedule_fa.py
demos the Exo scheduling-pass methodology end-to-end (commit 694611e
in main + d21ad05 in the exo fork). Then suggest a starting move.
```

---

## What the next session has to work with

- **Main repo** at `./` on `main`, 8 commits ending in
  `8e0dd19` (kickoff refresh).
- **Exo fork** at `./exo/` on `main`, 3 commits ending in
  `d21ad05` (vle/vse @instrs + vfexp body fix).
- **gem5 fork** at `./gem5/` on `stable`, commit `841d376`
  (unchanged this session).
- **Memory** at `~/.claude/projects/-home-noah-project-riscv/memory/`
  with the project-state record + 7 feedback memories including
  the new `feedback_exo_scheduling_idioms.md` from this session.

## Recommended openers by intent

- *"Scale the Exo scheduling pass toward Mo 8 cycle parity"* →
  start in `paper/exo_schedule_fa.py` and `paper/y1_handoff_kickoff.md`
  §"Notes for the Mo 8 follow-on". The 4 concrete sub-steps are in
  the kickoff under Y2 work item 8.
- *"Push public repos"* → user-action; assistant can prep by
  proposing a real LICENSE copyright line + final-state diff
  audit, but cannot `git push` without the user's hosting choice
  and credentials.
- *"Submit GCC bug reports"* → user-action; `paper/gcc_bug_report.md`
  is the polished draft.
- *"Rewrite advisor pitch"* → `paper/advisor_msg.txt` is stale
  on the abandoned Path A; assistant can draft v3 around the
  Y1 results (workshop submission + open-source artifact +
  Mo 8 foundation + FireSim Y2 plan).
- *"Polish §5.5 to reflect the Mo 8 foundation"* → the paper draft
  currently calls the precision-routing pass a "Y2 deliverable";
  small edit since the foundation now exists.

## Operating mode (preserved from prior memory; in
`~/.claude/projects/-home-noah-project-riscv/memory/`)

- `feedback-direct-recommendations`: give the technical playbook
  with reasoning, not scope-narrowing.
- `feedback-keep-work-dir-clean`: ephemeral scratch → /tmp,
  durable artifacts → `./paper/`.
- `feedback-rvv-asm-vsetvli`: disassemble + check vsetvli stream
  before believing operand-width hypotheses.
- `feedback-fu-stub-brackets`: when projecting a HW FU on a
  non-HW simulator, bracket conservative + aggressive stubs.
- `feedback-distance-based-goldens`: for FP-conversion Chisel
  tests, build the golden as distance search + exhaustive sweep.
- `feedback-exo-scheduling-idioms` (NEW): replace() matches loops
  by iter-var name; Exo's unification refuses @instrs whose
  bodies don't reference all declared parameters.

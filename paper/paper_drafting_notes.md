# Paper drafting notes (NOT for submission)

Working notes for `paper_draft.md`. Lives separately so the paper file
itself contains only paper-bound prose.

Current snapshot: **8 of 9 sections in DRAFT** (Abstract, §1 Intro,
§2 Background, §4 Hardware, §5 Compiler, §6 Kernel, §7 Evaluation,
§8 Related Work, §9 Conclusion). Only §3 Per-Stage Precision
Analysis remains as OUTLINE, gated on the perplexity sweep.

## Numbers still needed before submission

- **E2E Llama-3.2-1B decode tok/s** — pending FireSim FPGA path or HBM
  model in gem5. Y2 work; Y1 paper frames this as projected.
- **Perplexity** (Llama-3.2-1B, WikiText-2 + C4) — pending NVFP4
  calibration via ARCQuant tooling. Gates §3.2; needs GPU + model +
  calibration setup.
- **Compiler-parity number** (Exo-generated vs hand-coded) — pending
  the precision-routing scheduling pass implementation against the
  §6 kernel. Y2 work.

## Sections to draft next

In priority order (most data-ready first):

1. **§3 Per-Stage Precision Analysis** (~500 words). Inputs in
   `precision_config.md`; gated on running the per-stage perplexity
   sweep on Llama-3.2-1B (§3.2). Needs GPU + ARCQuant fork.

After §3, paper is feature-complete narrative. Then enter the polish
loop: citation standardization, abstract trim, figure generation (FU
block diagram from `fu_sketch.md` → camera-ready SVG, comparison-table
formatting, etc.).

## Implementation status (cross-ref to paper §s)

| Component                              | Status                          | Paper § |
|----------------------------------------|---------------------------------|---------|
| Saturn RTL: 4 custom lanes + PolyExp   | DONE (57/57 ChiselTests)        | §4      |
| Yosys synth + 16 nm area estimate      | DONE                            | §4.7    |
| gem5 5.1 decoder + FU-latency patch    | DONE (4 lanes wired, 13.05c/iter verified) | §4.6 + §7 |
| Saturn fork upstream PR                | NOT FILED — pending repo cleanup | —       |
| Mixed-prec FA kernel (`native` bench)  | DONE                            | §6 + §7 |
| L-sweep cycle measurement (L2K-L16K)   | DONE                            | §7.5    |
| All-FU sidebar variant (`_allfu`)      | DONE                            | §7.5 sidebar |
| GCC vsetvli bug report                 | DRAFTED (`gcc_bug_report.md`)   | §7.7    |
| GCC bug submitted to bugzilla          | NOT YET                         | —       |
| Exo `@instr` library + memory class    | DONE (`exo/src/exo/platforms/saturn_rvv.py`) | §5      |
| Exo BF16 first-class patch             | DEFERRED (uint16 carrier in use) | §5.2    |
| Exo Mo-8-equivalent schedule program   | Y2 work                         | §5.6    |
| FireSim integration                    | Y2 work                         | §9      |
| llama.cpp E2E (Llama-3.2-1B)           | Y2 work (target speedup TBD)   | §9      |
| Perplexity sweep                       | NOT STARTED                     | §3.2    |
| hd=128 sweep                           | L2 K spot-check DONE 2026-05-17 — disproves "larger hd helps" hypothesis; ratio widens to 1.96× | §7.6    |

## Risk flags carried forward (verify before submission)

- **"Caught up" framing**: re-grep upstream llama.cpp at submission
  time. The closed-gap delta from `llamacpp_survey.md` was correct
  as of 2026-05-16; re-verify the `ggml_v_expf_m2` + SpacemiT FA
  claims periodically.
- **"First" implicit claims**: set arXiv keyword alerts for Benini /
  Ragan-Kelley / NVIDIA NVFP4 / Berkeley gemmini-mx groups. VMXDOTP
  + VEXP sequel is the biggest concurrent threat.
- **Open-source artifact public-repo deadline**: must precede camera-
  ready. Saturn fork, gem5 fork, Exo fork, kernel benches all need
  public-clean state.
- **≥ 1.5× E2E speedup vs K1**: needs a real K1 baseline (advisor
  borrow abandoned). FireSim FPGA or published K1 specs only.
- **GCC bug**: confirm reproducer on newer GCC (mainline / 14 / 15)
  before submitting bugzilla — current `gcc_bug_report.md` notes
  this caveat but the reporter hasn't tested newer versions.

## Camera-ready polish queue (deferred from internal read-through)

- Repetition of "≥1.5× not met" message across abstract / §1 / §7.5 /
  §7.6 / §9 — could compress to one strong statement in §7.6 with
  glancing refs elsewhere. (5 mentions currently.)
- Abstract length is ~280 words; workshop conventions are 200-250.
- Citation format mostly consistent ([Author, Venue'YY]) but a few
  bracket bare arXiv IDs — standardize.
- §6.1 pseudocode uses `e4m3_decode[]` lookup notation; real-Saturn
  path uses `vfconv.fp8.bf16.v`. Either add a footnote or generalize
  to `dequant_e4m3()` in pseudocode.
- Figure generation: §4 wants a FU block diagram (from
  `fu_sketch.md`); §6 wants a kernel dataflow figure; §7 wants a
  cycle-stack bar chart per variant.

# Survey: Alternative Attention HW Directions for Arch+Compiler PhD Signal

Scope: side-project, 12-month workshop-paper target (MLArchSys/EMC2), full-stack
(HW + compiler + ASIC + SW) hard requirement. Baseline = Saturn `vfexp` + fused
flash-attention RTL + MLIR FA pass + llama.cpp bench.

---

## 1. Block-Sparse Attention RTL on Saturn

**(a) OSS state (May 2026).**
- RTL: zero. Saturn (github.com/ucb-bar/saturn-vectors) has no sparse-attention
  code; Gemmini master/`gemmini-mx` has no sparse paths; Ara/T1/Vortex are dense
  vector only. SystolicAttention/FSA (arXiv:2507.11331) is closed.
- Compiler: MLIR `sparse_tensor` dialect lowers from `linalg` but is targeted at
  CSR/CSC/COO unstructured sparsity, *not* the block-structured BigBird/Longformer
  patterns; no pattern-match for attention masks. Triton `block_sparse` kernels
  exist (GPU only). XAttention (ICML'25, mit-han-lab/x-attention) and
  Block-Sparse-Attention (mit-han-lab/Block-Sparse-Attention) ship CUDA kernels.
- Software: llama.cpp has no sparse-attention path. DeepSeek NSA is software-only.

**(b) Recent papers.**
- Native Sparse Attention (NSA), arXiv:2502.11089, Feb 2025, DeepSeek/PKU/UW —
  hardware-aligned trainable sparse attention; GPU only.
- XAttention, ICML 2025, MIT HAN Lab — block-sparse with antidiagonal scoring.
- FSA / SystolicAttention, arXiv:2507.11331, 2025 — closed-source systolic.
- SWAT (arXiv:2405.17025, DAC'22, Bai/Mitra NUS) — FPGA sliding-window only.

**(c) Gap.** No open RTL implements NSA/XAttention-style hierarchical block
sparsity on an RVV core, and no MLIR pass maps `linalg.generic` + block-sparse
mask to sparse RVV intrinsics.

**(d) Faculty alignment.**
- **Song Han (MIT EECS)** — direct extender; XAttention, DuoAttention,
  Block-Sparse-Attention, LServe are all his. Strong signal but crowded lab.
- **Tushar Krishna (GT)** — SpAtten (HPCA'21), Sanger; sparse-attention
  systolic dataflow is his lane.
- **Mohamed Abdelfattah (Cornell Tech)** — BitMoD (HPCA'25), NSF CAREER on
  FPGA distributed LLM inference; would value RVV+block-sparse co-design.

**(e) Compiler co-design strength.** Strong. Pattern-matching `linalg.generic`
with mask attributes to a sparse-attention op is genuinely novel; tiling +
mask-aware scheduling + cost model for variable-density blocks is a meaty
problem. Probably the strongest compiler story of the three.

**(f) Scope (12 mo).** Tight but doable if you fix the sparsity *pattern*
(e.g., Mistral sliding-window OR Longformer global+local — pick one). Risk
explodes if you try to generalize. RTL: sparse index-stream FU + masked
vector-load on Saturn. Compiler: MLIR lowering pass for one mask family.
Eval: prefill-only on a Mistral-7B-like model.

**(g) Risks.**
- Concurrent: Song Han's lab will ship more block-sparse kernels (high pace).
  Beat them by being the *open RVV+RTL+MLIR* version, not by competing on
  algorithms.
- Sparsity pattern lock-in — pick the right one (sliding-window is safest).
- Mask-generation overhead can dwarf savings at workshop-paper scale; need a
  pure static-pattern baseline first.

**(h) vs baseline.** **Comparable, slightly stronger on compiler signal,
weaker on cleanliness.** Baseline's `vfexp` story is a tighter HW contribution;
block-sparse's compiler story is richer. If the advisor's compiler co-design
focus is the dominant signal, this is stronger. If RTL-tape-out-ready is the
signal, baseline wins.

---

## 2. On-Chip KV-Cache Compressor / Eviction Accelerator

**(a) OSS state (May 2026).**
- RTL: Kelle (arXiv:2510.16040, MICRO'25, Xia & Zhang NYU) ships a 32x32 RSA +
  SFU in SystemVerilog but the RTL is **not publicly released** as of May 2026.
  Oaken (arXiv:2503.18599, ISCA'25, KAIST Park lab) is also closed RTL on an
  internal LPU. NVIDIA kvpress is SW only. Longhorn Silicon (UT Austin) is
  planning RTL freeze for 2027 — not yet usable.
- Compiler: zero cache-aware tiling for RTL targets. vLLM's PagedAttention is
  the closest scheduler analog. Triton has SW kernels. No MLIR pass for
  KV-quant + eviction co-scheduling.
- Software: llama.cpp has `--cache-type-k q4_0 q8_0` paths but no
  importance-eviction. KIVI/KVQuant/CAKE are PyTorch only.

**(b) Recent papers.**
- Kelle, arXiv:2510.16040, MICRO 2025 — eDRAM + RSA co-design, edge.
- Oaken, arXiv:2503.18599, ISCA 2025 — online/offline hybrid quant, KAIST.
- KVQuant, NeurIPS 2024 — Shao group (Berkeley) algorithm side.
- Expected Attention (arXiv:2510.00636) — future-query-based eviction.
- CAKE (arXiv:2503.12491) — cascading eviction with layer preferences.
- InfiniGen, OSDI'24 (SNU, Sim) — dynamic KV management.

**(c) Gap.** No open RTL for a KV-cache importance scorer + quantizer + evictor
that integrates with an open vector/systolic core. The closest open thing is
*paged* allocator SW; the hardware-importance-scoring datapath is missing
entirely from open repos.

**(d) Faculty alignment.**
- **Yakun Sophia Shao (Berkeley)** — KVQuant first-authored from her group;
  Gemmini lineage; this is the single best PI-fit for this direction.
- **Jongse Park (KAIST → applicable for collaborations / not US PhD)** — skip
  for PhD targeting but cite Oaken.
- **Mohamed Abdelfattah (Cornell)** — KV/FlashDLM at ICLR'26; NSF
  CAREER on FPGA-distributed inference covers cache hierarchies.
- **Tushar Krishna (GT)** — communication/memory NoCs, perfect for the
  bandwidth side.

**(e) Compiler co-design strength.** Moderate. The compiler problem is
*scheduling* (cache-aware tiling, when-to-evict policies as ILP/heuristic) and
*lowering* (quant-dequant fusion into attention). Less pattern-matching, more
scheduling+cost-modeling. Decent for a co-design narrative but less
"compiler-first" than block-sparse.

**(f) Scope (12 mo).** Aggressive. KV scope creep is real: prefill vs decode,
quant kernels, eviction policies, page allocator, all interact. To fit
12 months: pick **decode-only**, **fixed-policy attention-score-based
eviction**, **INT4 group quant**, integrate against Saturn. Skip eDRAM
modeling (Kelle's wedge), skip multi-tenant.

**(g) Risks.**
- **Highest concurrent-paper risk** of the three. Kelle/Oaken/Tender are
  recent; NVIDIA, Longhorn, vLLM, NeurIPS workshops are crowded.
- Memory-system complexity — your eval needs realistic memory models (DRAM
  timing, HBM bandwidth). Without that, the paper isn't credible.
- Closed-IP dependency: comparing to Oaken/Kelle without their RTL is hard.

**(h) vs baseline.** **Higher ceiling, higher risk, lower fit for vector unit
side-project.** This direction is genuinely a memory-system + serving paper,
not a vector-ISA paper. If you have access to a Gemmini DSE + memory-system
infra (Berkeley/Chipyard), it could be excellent. As a solo applicant
side-project on Saturn, it's a stretch. **Weaker than baseline for the
applicant's constraints, stronger only for a Shao-aligned applicant who has
clear infra runway.**

---

## 3. Attention-Sink / StreamingLLM HW Support

**(a) OSS state (May 2026).**
- RTL: SWAT FPGA (arXiv:2405.17025, NUS, DAC'22) is the closest published
  RTL — sliding-window only, FPGA dataflow, no Saturn/Gemmini integration,
  no sink-token preservation. No open RTL implements sink+window jointly.
- Compiler: zero. No MLIR/Triton/TVM pass for streaming attention schedule
  (block-wise sink concat + circular window pointer).
- Software: mit-han-lab/streaming-llm is the ICLR'24 reference. DuoAttention
  (ICLR'25) integrates retrieval+streaming heads. Sink mechanism is now in
  TensorRT-LLM and OpenAI's gpt-oss release (Aug 2025). llama.cpp has *no*
  attention-sink path as of May 2026.

**(b) Recent papers.**
- StreamingLLM, ICLR 2024, arXiv:2309.17453 — Xiao/Tian/Chen/Han/Lewis.
- DuoAttention, ICLR 2025, mit-han-lab — retrieval+streaming dual cache.
- SWAT, ICLR/preprint 2025 (arXiv:2502.18845) — sigmoid+ALiBi to fix sink
  variance; training-time fix.
- LServe, MLSys-track 2025, mit-han-lab — unified sparse-attention serving
  (folds sink+window into one runtime).

**(c) Gap.** No open RVV intrinsic / RTL FU that does
sink-preservation + circular-buffer window + sink-aware causal mask in a
single op, and no MLIR pass emits it. The HW story is small (basically a
circular pointer + an index-list FU) but it's the *cleanest* full-stack story.

**(d) Faculty alignment.**
- **Song Han (MIT)** — owns this entire space; DuoAttention, LServe, the
  original StreamingLLM. PhD admissions: highly competitive, may not value a
  derivative of his own work. Better as a *cited reference* in your SOP than a
  PI target.
- **Vijay Reddi (Harvard)** — embodied/edge inference, MLPerf; would value
  long-context-on-edge.
- **Tushar Krishna (GT)** — sliding-window dataflow is exactly his MAERI/STONNE
  lineage.

**(e) Compiler co-design strength.** Weakest of the three. The "compiler"
problem is essentially: emit a streaming-schedule that reuses the window
buffer + concatenates sink tokens. Doable as an MLIR pass but it's mostly
*loop transformation*, not pattern-matching/cost-modeling. Not a meaty
compiler thesis.

**(f) Scope (12 mo).** Best fit of the three for solo 12-month pace. HW is
minimal (sink-aware mask FU + circular pointer), SW is small (llama.cpp PR
for sink path + ~1 MLIR pass), eval is clean (long-context perplexity +
throughput vs window). Concrete deliverable in 6-8 months.

**(g) Risks.**
- **Algorithm is too well-known.** Reviewers will ask "what's new?" Answer
  must be: first open RVV+RTL+MLIR end-to-end, with a numerical contribution
  (e.g., sink-aware fp8 scaling, or quantized sink cache).
- TensorRT-LLM and gpt-oss have already absorbed sinks → industry view is
  "solved." Need to angle as edge/RISC-V, not server.
- Song Han's lab can ship anything in this space faster than you can.

**(h) vs baseline.** **Weaker as a research contribution, comparable as an
engineering deliverable, weaker on compiler signal.** This is the lowest-risk
direction to *finish* in 12 months, but the highest risk that reviewers say
"so what." Reasonable as a *complement* (added feature to baseline FA
implementation) but not a standalone replacement.

---

## Bottom line

| Direction | HW novelty | Compiler novelty | Scope risk | Concurrent risk | Faculty fit |
|---|---|---|---|---|---|
| Baseline (`vfexp`+FA+MLIR) | Med | Med | Low | Low | Broad |
| 1. Block-sparse | Med-High | **High** | Med | Med-High | Han/Krishna/Abdelfattah |
| 2. KV-cache RTL | High | Med | **High** | **High** | Shao (strong), Krishna, Abdelfattah |
| 3. Sink/Streaming | Low-Med | Low | Low | Med | Han, Krishna |

**Recommendation:** Baseline + sink-token feature dropped in as a bonus
section is the safest 12-month deliverable. If you want stronger
compiler-co-design signal, **direction 1 (block-sparse, Mistral sliding-window
pattern, open RVV+MLIR)** is the best swap — it strictly dominates baseline
on compiler signal while keeping a comparable HW scope, and it aligns with
the largest set of target PIs (Han, Krishna, Abdelfattah, plus Shao via
sparsity overlap). **Direction 2 is the highest-ceiling thesis pitch** but is
the wrong fit for a Saturn-based, solo, 12-month side-project; pursue it only
if first-year PhD infra (Gemmini + Chipyard memory-system + a co-author) lands.

**Sources** (verifiable, May 2026): arXiv:2502.11089 (NSA), arXiv:2510.16040
(Kelle), arXiv:2503.18599 (Oaken), arXiv:2405.17025 (SWAT), arXiv:2309.17453
(StreamingLLM), arXiv:2502.18845 (SWAT-train), arXiv:2507.11331
(SystolicAttention), mit-han-lab/{streaming-llm, x-attention, duo-attention,
Block-Sparse-Attention}, ucb-bar/{saturn-vectors, gemmini},
abdelfattah-lab/BitMoD-HPCA-25, hanlab.mit.edu/projects/streamingllm,
csl.cornell.edu/~zhiruz, tusharkrishna.ece.gatech.edu,
people.eecs.berkeley.edu/~ysshao, mohsaied.com.

# Survey: System-Level Alternative Directions vs. Saturn `vfexp` + MLIR FA Baseline

Three system-level directions for an LLM-accel PhD application side-project, evaluated for arch+compiler co-design signal, OSS gaps, and feasibility at a 12-month workshop-paper pace. All Web evidence cross-referenced May 2026.

---

## Direction 1: MoE Routing / Expert Dispatch Accelerator

**(a) OSS state (May 2026):**
- **RTL:** No MoE-specific RTL exists in Chipyard, ESP, OpenPiton, Vortex, or FireSim. Edge-MoE (FPGA, vision ViTs) is the closest open RTL but is not LLM-targeted and not Chipyard-integrated.
- **Compiler:** MLIR has no MoE-aware dispatch lowering. TVM has rudimentary MoE patterns; no scheduling pass for token-to-expert assignment on accelerators.
- **Software:** vLLM v0.9+ supports Expert Parallelism with EPLB load balancer (docs.vllm.ai/serving/expert_parallel_deployment); SGLang has DeepSeek-V3 EP day-one. **Neither runs on RISC-V.** llama.cpp has CPU/CUDA MoE; no RVV path, no MoE on Saturn.

**(b) Recent papers:**
- **EARTH** (ASPLOS '26, Mar 2026, doi:10.1145/3779212.3790155) — MoE accelerator with entropy-aware speculative prefetch + dual-entropy encoding + result reuse. ~2x energy reduction.
- **MoE-SpeQ** (arXiv 2511.14102, Nov 2025) — speculative quantized decoding + proactive expert prefetching for MoE inference.
- **SP-MoE** (arXiv 2510.10302, Oct 2025) — speculative decoding + prefetching for MoE.
- **LayerScope** (arXiv 2509.23638, Sep 2025) — predictive cross-layer MoE scheduling.

**(c) OSS gap (one sentence):** No open-source RISC-V SoC has a hardware top-k router + crossbar expert-dispatch unit with a corresponding MLIR/llama.cpp dispatch pass; existing MoE accelerator papers (EARTH, UbiMoE) are simulator-only or FPGA-vision-only.

**(d) Faculty alignment:**
- **Tushar Krishna (GT)** — Synergy Lab MoE/sparse work (FlexInfer EuroMLSys'25, FlexiBit ISCA'25 mixed precision). Routing/dispatch interconnect is core to his NoC research lineage.
- **Sophia Shao (Berkeley)** — Gemmini/Chipyard. Recent FGMP (mixed-precision LLM, '25), KVQuant (NeurIPS '24). Would advise the RTL integration but MoE is not her direct focus.
- **Mohamed Abdelfattah (Cornell)** — BitMoD (HPCA '25) algorithm-HW co-design for LLM; FPGA/datatype focus; MoE compatible.

**(e) Compiler co-design strength:** **Very high.** Dispatch is a scheduling problem first, a hardware problem second: batch packing, expert prefetch prediction, capacity-factor lowering, expert-parallel partitioning. MLIR pass writing here is more meaningful than the FA-lowering pass in the baseline.

**(f) Realistic 12-month scope:** **Tight.** Realistic subset: (1) Top-k routing unit + crossbar in Chisel on Chipyard (3 mo); (2) Mixtral-8x7B or Qwen-MoE-A3B end-to-end via custom llama.cpp MoE kernel calling the unit (4 mo); (3) MLIR-based MoE expert-dispatch lowering pass (3 mo); (4) FireSim eval + paper (2 mo). Cut: skip full expert weight movement engine; assume on-chip weights or simple DMA.

**(g) Risks:**
- Scope creep: capacity factor handling, dropped tokens, expert imbalance can each be a paper.
- Demonstration: needs an actual MoE running end-to-end; if expert weights don't fit, you need a memory hierarchy story (and you're adding a research direction).
- Competition: EARTH already exists at ASPLOS '26. Differentiation requires either the **RISC-V + open-source angle** (industry can't do this) or a specifically novel dispatch/scheduling primitive.

**(h) vs. baseline:** **Stronger if you can finish it.** MoE dispatch is a true arch+compiler co-design problem with a healthy compiler dimension; FA-lowering is mostly a compiler exercise on a fixed primitive. But baseline is ~60% lower scope; this direction risks an unfinished demo. **Verdict: stronger ceiling, weaker floor.**

---

## Direction 2: Speculative Decoding Accelerator

**(a) OSS state (May 2026):**
- **RTL:** No open-source spec-decode-specific RTL on Chipyard or any RISC-V SoC. HADES (below) is the only academic HW spec-decode work and it is FPGA/HLS-only, no public RTL.
- **Compiler:** No MLIR/TVM speculation-depth scheduling pass. P-EAGLE (vLLM v0.16+, Dec 2025) is Python-level only. Speculators v0.3 (blog.vllm.ai/2025/12/13) is a training framework, not a compiler.
- **Software:** vLLM v1, SGLang (1.8x via MTP), TensorRT-LLM all ship EAGLE-3 / Medusa / N-gram drafters. SpecForge (lmsys.org/blog/2025-07-25) trains drafters for SGLang. **None on RISC-V.** llama.cpp has spec-decode in CPU/CUDA; no RVV.

**(b) Recent papers:**
- **HADES** (arXiv 2412.19925, Dec 2024) — only HW spec-decode paper; OpenHLS + Vivado FPGA + Synopsys VCS ASIC sketch. Authors Z. Yang, Y. Jin, X. Xu (not from a top arch lab).
- **EAGLE-3** (NeurIPS '25) — SW algorithm, 2-6x speedup; the SOTA draft model.
- **PEARL** (ICLR '25) — parallel speculative decoding, adaptive draft length.
- **Dovetail** (arXiv 2412.18934) — CPU/GPU heterogeneous spec decode.
- **Speculative Speculative Decoding** (arXiv 2603.03251, 2026) — multi-level drafting.

**(c) OSS gap (one sentence):** No open RISC-V SoC has a parallel verification unit + draft-model coresidency with a corresponding compiler pass for speculation-depth scheduling and draft dispatch.

**(d) Faculty alignment:**
- **Song Han (MIT)** — TLT speculative-RL training '25, broader efficient inference. Strongest LLM-side alignment; less RTL focus.
- **Sophia Shao (Berkeley)** — would advise the Chipyard integration; closer to FGMP/KVQuant scope.
- **Tri Dao (Princeton)** — Mamba/FlashAttention; speculation algorithms are adjacent.
- **Mohan Kumar / Hadi Esmaeilzadeh, Christos Kozyrakis (Stanford)** — SGLang origin Kozyrakis would care about full-stack.

**(e) Compiler co-design strength:** **Medium-high.** Real compiler problems: scheduling target verification vs. draft generation, KV-cache-shared layouts, branch tree management (EAGLE-3 dynamic tree). Pure-compiler papers on this exist (NVIDIA Dynamo, Speculators), so the **hardware angle is the differentiator.** Hardware is most natural at parallel verification — vector-unit-side multi-token softmax + sample.

**(f) Realistic 12-month scope:** **Most defensible.** Path: (1) Saturn vector unit extension for parallel verification (multi-token softmax + sampling) — 2 mo; (2) Co-resident draft (smaller Llama) + target on same SoC with shared KV — 3 mo; (3) llama.cpp speculative path on RVV — 3 mo; (4) MLIR/scheduler pass for spec depth — 2 mo; (5) Eval + paper — 2 mo. **Smallest moving parts; vector-unit extension is concrete.**

**(g) Risks:**
- TensorRT-LLM and vLLM dominate spec-decode on GPU; positioning is "open-source, RISC-V, full-stack" not "faster than NVIDIA."
- Algorithmic frontier (EAGLE-3, MTP, dynamic trees) moves quarterly — pick a fixed target.
- HADES exists but is weak; not a real defended baseline.
- End-to-end demo needs a real second model fitting on FPGA.

**(h) vs. baseline:** **Comparable/orthogonal.** Spec-decode is *less* of a HW novelty (verification ALU is simpler than fused FA) but *more* of a system-level scheduling story. Better PhD narrative: "I co-designed HW + sched + compiler for a true inference-time bottleneck," vs. baseline's "I built a fused kernel." **Verdict: orthogonal; arguably better arch+compiler narrative, slightly weaker pure-HW novelty.**

---

## Direction 3: Prefill / Decode Disaggregation On-Chip

**(a) OSS state (May 2026):**
- **RTL:** **No open RTL.** SPAD (below) is the only HW disaggregation paper, simulator-only. Nothing in Chipyard, ESP, FireSim showcases. NVIDIA Dynamo (GTC '25), Meta production, llm-d, vLLM disagg-prefill are all software/multi-GPU, not on-chip heterogeneous.
- **Compiler:** vLLM disagg-prefill experimental feature (docs.vllm.ai/features/disagg_prefill) is request-routing, not a compiler. No MLIR phase-aware lowering pass.
- **Software:** vLLM, SGLang (PD-disagg tuned for MoE), llm-d, Together CPD all support PD-disagg at the cluster level. **Zero on-chip / single-SoC implementations.**

**(b) Recent papers:**
- **SPAD** (arXiv 2510.08544, Oct 2025) — Princeton/UW (Zhang, Patel, Ning, Wentzlaff): specialized prefill chip (large systolic, GDDR7) + decode chip (smaller compute, HBM3). Simulator only. 19-41% cost, 2-17% TDP savings.
- **Nexus** (arXiv 2507.06608, Jul 2025) — intra-GPU SM-partition PD-disagg.
- **DuetServe** (arXiv 2511.04791, Nov 2025) — fine-grained SM-level phase isolation.
- **Trinity** (arXiv 2512.02281, Dec 2025) — disagg vector search for RAG.
- **CXL-SpecKV** — CXL-disagg KV cache FPGA work.

**(c) OSS gap (one sentence):** No open-source heterogeneous RISC-V SoC integrates a prefill engine (compute-heavy) + decode engine (BW-heavy) on one die with a phase-aware MLIR scheduling pass — SPAD is a simulator paper, the entire OSS RTL ecosystem is blank.

**(d) Faculty alignment:**
- **David Wentzlaff (Princeton)** — SPAD author. Direct extension.
- **Pratyush Patel (UW)** — SPAD co-author; systems/architecture for ML serving.
- **Sophia Shao (Berkeley)** — Gemmini/Chipyard heterogeneous SoC expertise; natural fit for HW integration.
- **Christos Kozyrakis (Stanford)** — SGLang co-author, LLM serving; weaker on RTL.
- **Mohan Kumar / Mark Hill (Cornell?)** — heterogeneous SoCs.

**(e) Compiler co-design strength:** **Highest of the three.** Phase detection, KV-cache transfer scheduling between engines, dynamic batching across phases, attention-vs-matmul dispatch — *all* compiler/runtime concerns. The compiler/scheduler dimension dominates the HW dimension.

**(f) Realistic 12-month scope:** **Tightest.** This is genuinely a 2-3 year project at PhD-level intensity. Achievable subset: (1) Two Gemmini instances (one configured for prefill: large array, one for decode: small array) on one Chipyard SoC + Saturn for attention — 4 mo; (2) Phase-aware MLIR/runtime dispatch pass + custom llama.cpp glue — 4 mo; (3) Small-model end-to-end (TinyLlama / Llama-3.2-1B) on FireSim — 3 mo; (4) Paper — 1 mo. **Demo will be modest; positioning is "first open-source on-chip PD-disagg testbed."**

**(g) Risks:**
- **Highest scope risk.** SPAD itself is simulator-only because building it is hard.
- Two-engine + interconnect + KV transfer is genuinely SoC-engineering-heavy; one applicant can't deliver a competitive engineered chip.
- Demonstration: end-to-end LLM that meaningfully saturates both engines requires a real workload split.
- Mitigation: position as a **methodology / testbed contribution** ("first open RTL+compiler PD-disagg framework on Chipyard"), not a performance contribution.

**(h) vs. baseline:** **Larger ambition, weaker defensibility at workshop-paper scope.** Baseline is fully achievable and presents a complete narrative. Disaggregation is glamorous and PhD-faculty-aligned (SPAD authors are at target schools) but the realistic deliverable is "a testbed and a small demo" rather than "a result." **Verdict: bigger headline, but high risk of an undercooked demo. Choose only if the applicant accepts the testbed-not-result framing.**

---

## Bottom-line Comparison

| Direction | HW novelty | Compiler dimension | 12-mo feasibility | OSS gap clarity | Faculty hit-rate |
|---|---|---|---|---|---|
| **Baseline (Saturn `vfexp` + MLIR FA)** | Med (fused op) | Med | **High** | High (no RVV exp/softmax) | Med (Shao, Krste) |
| **1. MoE dispatch** | High | **High** | Medium | High | High (Krishna, Shao, Abdelfattah) |
| **2. Spec decode** | Med | High | **High** | High | High (Han, Shao, Dao) |
| **3. PD disagg** | High | **Highest** | Low-Med | Highest | Highest (Wentzlaff, Patel, Shao, Kozyrakis) |

**Recommendation framing for the applicant:**
- **Spec decode** is the strongest *swap* for the baseline — comparable scope, better arch+compiler narrative, more compiler-co-design surface area, exists in the same Saturn-vector-extension idiom.
- **MoE dispatch** is the strongest *upgrade* if the applicant is willing to accept higher scope risk; it has the strongest compiler co-design + faculty alignment combo.
- **PD-disagg** is the most prestigious headline but is the least defensible at side-project pace; only attempt if reframing as "first open testbed" is acceptable.
- **Baseline** remains the safest deliverable and is not obviously dominated; the question is whether the applicant wants safety or upside.

---

## Sources

- EARTH (ASPLOS '26): https://dl.acm.org/doi/abs/10.1145/3779212.3790155
- MoE-SpeQ: https://arxiv.org/abs/2511.14102
- SP-MoE: https://arxiv.org/abs/2510.10302
- LayerScope: https://arxiv.org/html/2509.23638
- HADES: https://arxiv.org/abs/2412.19925
- EAGLE: https://github.com/SafeAILab/EAGLE (EAGLE-3 NeurIPS '25)
- PEARL: https://github.com/smart-lty/parallelspeculativedecoding (ICLR '25)
- Speculators v0.3 (vLLM): https://blog.vllm.ai/2025/12/13/speculators-v030.html
- SpecForge / SGLang: https://www.lmsys.org/blog/2025-07-25-spec-forge/
- SPAD: https://arxiv.org/abs/2510.08544
- Nexus: https://arxiv.org/abs/2507.06608
- DuetServe: https://arxiv.org/pdf/2511.04791
- vLLM disagg-prefill: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM EP: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- Chipyard: https://github.com/ucb-bar/chipyard
- Saturn: https://saturn-vectors.org/ and https://github.com/ucb-bar/saturn-vectors
- Krishna / Synergy: https://synergy.ece.gatech.edu/, FlexiBit https://arxiv.org/abs/2411.18065
- Shao / SLICE: https://slice.eecs.berkeley.edu/
- Wentzlaff: https://www.princeton.edu/~wentzlaf/
- Abdelfattah / BitMoD: https://github.com/abdelfattah-lab/BitMoD-HPCA-25

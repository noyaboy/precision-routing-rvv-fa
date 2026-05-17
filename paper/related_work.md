# Related-Work Foundation: Mixed-Precision Fused FA on Saturn + NVFP4 + Exo
*Targeted at MLArchSys @ ISCA 2027. Compiled May 2026; **rewritten 2026-05-16 PM** for merged X+Y direction.*

This document fills three specific gaps not covered by the running survey: (1) the FlashAttention algorithmic lineage, (2) the hardware attention-accelerator landscape, (3) the RVV custom-instruction / FU-extension ecosystem. A 1-page intro outline follows.

**Framing shift 2026-05-16 PM:** Earlier draft framed contribution as "first open RVV fused FA + custom softmax-exp FU." Upstream llama.cpp now has both a polynomial RVV softmax (`ggml_v_expf_m2` in `vec.h:1342`) and SpacemiT's K1-tuned VLEN=1024 fused FA (`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1121, 1305`). The novelty hook has shifted to **per-stage mixed-precision routing** (NVFP4 K/V cache + FP8-E4M3 attention weights + BF16 logits + FP32 softmax sum + vfexp inner-loop FU), with four new RVV custom instructions co-designed with an Exo 2 precision-routing pass. vfexp is now one of four FUs, not the headline. See [[project-side-project-direction]].

---

## Gap 1 — FlashAttention Algorithmic Lineage

### Core line (Dao et al.)

**FlashAttention-1** [Dao 2022]. *arXiv:2205.14135*, NeurIPS 2022. Authors: Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford). **Contribution:** IO-aware exact attention via blockwise tiling + online softmax recurrence (Milakov-style), reducing HBM↔SRAM traffic. **RVV applicability:** *Very high.* The algorithm is hardware-agnostic — tiling fits any cache hierarchy with explicit movement; the running-max / running-sum recurrence is a pure scalar-+-vector reduction that maps cleanly to RVV `vfredmax` / `vfredsum`. The IO-aware framing applies to L1/L2/scratchpad on a RVV core as much as it does to SRAM/HBM on a GPU.

**FlashAttention-2** [Dao 2023]. *arXiv:2307.08691*, ICLR 2024. Author: Tri Dao. **Contribution:** Refactors FA-1 to (a) reduce non-matmul FLOPs (defers rescaling until the end of the inner loop), (b) parallelize attention across sequence dimension per head, (c) better intra-warp work partitioning. Reaches 50–73% of A100 peak. **RVV applicability:** *High and load-bearing.* The reduced-rescale rewrite is the reference algorithm for almost every downstream FA implementation; the sequence-dim parallelization translates directly to multi-core / multi-cluster RVV. This is the version Titopoulos arXiv:2510.06834 ports to gem5 RVV.

**FlashAttention-3** [Shah 2024]. *arXiv:2407.08608*, NeurIPS 2024. Authors: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. **Contribution:** Hopper-specific: (a) producer-consumer warp specialization for WGMMA/TMA asynchrony, (b) hiding softmax under async block-wise GEMM (the "pingpong" schedule), (c) FP8 with block quantization + incoherent processing. Hits ~75% H100 utilization. **RVV applicability:** *Partial.* The intra-tile pipeline-hiding idea (overlap softmax with next-tile matmul) translates if Saturn issues vector chaining or has decoupled FUs — but the producer-consumer warp model is GPU-specific. FP8 portion overlaps with our quantization plans.

**FlashDecoding** [Dao 2023b]. Stanford CRFM blog (Oct 12 2023), no arXiv. Authors: Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov. **Contribution:** Adds a parallelization axis along the **K/V sequence length** for the *decode phase* (Q-len = 1), where standard FA leaves the GPU starved. Splits keys into chunks → parallel partial softmax → log-sum-exp reduction. Up to 8× on long-context decode. **RVV applicability:** *High for inference workloads.* The split-K + final LSE merge is precisely what an edge RVV core wants for long-context KV-cache decode, where a single query vector reads a long K/V history.

**FlashDecoding++** [Hong 2024]. *arXiv:2311.01282*, MLSys 2024. Authors: Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang (Tsinghua / Infinigence). **Contribution:** (a) Asynchronous softmax with *unified max value* (eliminates inter-block max-update sync), (b) flat-GEMM optimization with double buffering for the matrix-vector decode case, (c) heuristic dataflow selection between tensor-core and CUDA-core paths. 4.86× over HF baseline. **RVV applicability:** *Moderate.* The unified-max trick requires a calibration pass per model; the flat-GEMM optimization is heavily GPU-specific. The decode-side scheduling philosophy is useful as inspiration only.

**FlashInfer** [Ye 2025]. *arXiv:2501.01005*, MLSys 2025 (Best Paper). Authors: Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, Luis Ceze. **Contribution:** Customizable attention engine for *serving* — block-sparse + composable KV cache layouts, JIT-compiled attention templates, load-balanced scheduling compatible with CUDAGraph. **RVV applicability:** *Low-direct, high-conceptual.* The "templated attention kernel + JIT" pattern is exactly the Exo `@instr` philosophy at a higher level of abstraction; FlashInfer makes a strong case that *no single attention kernel is right* — a position the applicant should adopt.

### Quantization / low-precision variants

**SageAttention** [Zhang 2024]. *arXiv:2410.02367*, ICLR 2025. Authors: Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen (THU-ML). **Contribution:** Plug-and-play INT8 attention via per-token (Q,K) quantization + smoothing; FP16 accumulator. ~2.1× over FA2 on RTX4090. **RVV applicability:** *Moderate.* RVV widening MACs (`vwmacc`) accumulate INT8 → INT32 natively; the smoothing transform is a one-shot offline calibration.

**SageAttention2** [Zhang 2024b]. *arXiv:2411.10958*. **Contribution:** Per-thread INT4 for QK, FP8 for PV, with thorough outlier smoothing. 481 TOPS on 4090 (3× FA2). **RVV applicability:** Limited — INT4 mma is not in RVV 1.0; would need a custom Saturn FU.

**SageAttention3** [Zhang 2025]. *arXiv:2505.11594*. **Contribution:** Microscaling FP4 (MXFP4) attention for inference, plus 8-bit training exploration. **RVV applicability:** Very limited without dedicated FP4 hardware; aligns with MicroScopiQ direction.

### Long-context / distribution variants

**Ring Attention** [Liu 2023]. *arXiv:2310.01889*. Authors: Hao Liu, Matei Zaharia, Pieter Abbeel (Berkeley). **Contribution:** Blockwise FA across devices arranged in a ring, with KV-block rotation overlapped with compute. Enables near-infinite-context training. **RVV applicability:** *Low* for single-core edge work but *relevant* for multi-cluster Saturn deployments and a natural extension story.

**BurstAttention** [Sun 2024]. *arXiv:2403.09347*. Authors: Sun, Han, Liu et al. **Contribution:** Two-level partition (inter-device + intra-device) with Global/Local Attention Optimizations, reducing communication 40% vs Ring on 32×A100 at 128K context. **RVV applicability:** Same as Ring — multi-cluster only.

**Striped Attention** [Brandon 2023]. *arXiv:2311.09431*. Better load balancing for *causal* Ring Attention by interleaving sequence stripes rather than contiguous blocks. Worth a citation but not central.

### Architectural variants (model side, but they reshape the attention kernel)

**MQA** [Shazeer 2019]. *arXiv:1911.02150*. Single shared K/V head → ~8× decode bandwidth reduction; quality regression unless retrained.

**GQA** [Ainslie 2023]. *arXiv:2305.13245*, EMNLP 2023. Authors: Ainslie, Lee-Thorp, de Jong et al. (Google). **Contribution:** Group-shared KV — generalization of MQA; uptrain MHA→GQA cheaply. Standard in Llama-2/3, Mistral, Qwen. **Direct relevance:** Saturn's KV-cache RTL design choices must default to GQA arithmetic intensity (≈4–8 Q heads per KV head); this changes the optimal tile shape.

**Multi-Head Latent Attention (MLA)** [DeepSeek 2024]. *arXiv:2405.04434* (DeepSeek-V2 paper). **Contribution:** Compresses KV into a low-rank latent space; KV cache shrinks by ~10×. **RVV relevance:** Now mainline in DeepSeek-V2/V3/R1. Saturn's fused-FA design should either support an MLA path or be honest about being MHA/GQA-only. See also hardware-centric analysis arXiv:2506.02523 [Lin 2025].

### Critical analysis: which FA should the applicant implement?

**Recommendation: FA-2 forward + FlashDecoding for the decode phase, both as a single RTL pipeline with a runtime mode switch.** Reasoning:

1. **FA-2 is the *canonical reference*.** Every comparison paper (Titopoulos, FuseMax, SystolicAttention) reports against an FA-2-style algorithm. Picking FA-3 invites reviewer pushback ("Hopper-specific, not portable") and FA-1 invites "outdated, more FLOPs than necessary".
2. **FA-3 features don't help Saturn.** Warp-specialized async producer/consumer relies on hardware concepts (WGMMA, TMA) that simply don't exist in a RVV in-order or short-OoO core. The FP8 quantization aspect overlaps with the M2XFP / MicroScopiQ direction and should be argued as orthogonal.
3. **Decode coverage is non-negotiable for llama.cpp end-to-end.** Edge LLM inference is dominated by *decode*, and FA-2's per-query parallelization starves a single-core RVV. Adding the FlashDecoding split-K axis gives ~3–8× on long-context generation and is a one-paragraph extension to the FA-2 RTL.
4. **GQA-default arithmetic intensity** must be the design assumption — 95% of post-2024 open-weight LLMs ship with GQA.

Honest caveat: implementing both prefill (FA-2) and decode (FlashDecoding) in a *single* fused RTL is non-trivial — they have different parallelization axes and different KV-streaming patterns. The applicant should consider whether one RTL with a mode bit, or two specialized RTLs, is the cleaner story.

---

## Gap 2 — HW Attention Accelerators

### Pre-FuseMax

**A³** [Ham 2020]. *HPCA 2020*. Authors: Tae Jun Ham, Sungjun Jung, Seonghak Kim et al. (Seoul National University). **Architecture:** Approximate dot-product search engine + dedicated softmax / weighted-sum unit. Custom hash + early-termination on low-score K/V. **Axes:** Dense base + sparse Approx-A³ variant; FP; full attention (no prefill/decode split — pre-LLM era).

**ELSA** [Ham 2021]. *ISCA 2021*. Authors: Tae Jun Ham, Yejin Lee, Seonghak Kim et al. **Architecture:** LSH-based filter front-end + dense back-end. Hash-and-prune K/V before the dot product; 58× geomean over GPU. **Axes:** Approximate-sparse, FP, full attention. **Lineage:** A³ → ELSA is the same SNU group iterating on filter quality.

**SpAtten** [Wang 2021]. *HPCA 2021*. Authors: Hanrui Wang, Zhekai Zhang, Song Han (MIT HAN Lab). **Architecture:** Cascade token + head pruning, progressive quantization (MSB-first), top-k engine. **Axes:** Dynamic sparse (token + head), mixed-precision INT, prefill-focused. The first "open-source attention accelerator" people actually cite.

**Sanger** [Lu 2021]. *MICRO 2021*. Authors: Liqiang Lu, Yicheng Jin et al. (PKU). **Architecture:** Software prunes attention into a *structured* sparse pattern (mask predictor); hardware is a *reconfigurable* score-stationary systolic array supporting both SDDMM and SpMM. **Axes:** Dynamic structured sparse, FP, prefill.

**DOTA** [Qu 2022]. *ASPLOS 2022*. Authors: Zheng Qu, Liu Liu, Fengbin Tu, Zhaodong Chen, Yufei Ding, Yuan Xie. **Architecture:** Jointly-trained lightweight Detector predicts *weak* attention edges at runtime; main pipeline omits them. **Axes:** Predicted-sparse, FP, prefill. 152× over GPU.

**Energon** [Zhou 2022]. *HPCA 2022 / IEEE TC*. **Architecture:** Mix-Precision Multi-Round Filtering (MP-MRF) — low-precision filter pass identifies salient Q/K pairs, then high-precision recompute. **Axes:** Mixed-precision filter-then-refine, FP, prefill.

**FACT (ISCA 2023)** [Qin 2023]. *ISCA 2023*. Authors: Yubin Qin, Yang Wang, Dazheng Liu, Zhiren Zhao, Ying Wang, Yang Yang, Yuanqing Wang et al. **Note:** Not the Vivienne Sze paper the prompt's bullet referenced — this FACT is from CAS/ICT. **Architecture:** Eager-prediction co-design — predicts attention matrix *before* QKV generation, then mixed-precision FFN. **Axes:** Mixed-precision predicted-sparse + FFN co-opt, FP/INT, full transformer (not just attention). **Honesty flag:** I could not find a Vivienne-Sze-authored "FACT" paper at HPCA 2023 or MICRO 2023. The paper at this name is the CAS/ICT one; the Sze group's closest analog is the FLAT (Kao et al., ASPLOS 2023) dataflow paper, which is fused-attention dataflow analysis, *not* a chip.

**CTA** [Wang 2023]. *HPCA 2023*. Authors: Haoran Wang, Haobo Xu, Ying Wang, Yinhe Han. Compressed-token attention via semantic-feature deduplication; sparse + token compression at the input side.

**FLAT** [Kao 2023]. *ASPLOS 2023*. Authors: Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, Amir Yazdanbakhsh, Tushar Krishna (Google + Georgia Tech). **Architecture:** Fused-attention *dataflow* analysis on a Timeloop-modeled spatial array — not a built chip, but the methodology FuseMax compares against. **Axes:** Dense, FP, prefill, dataflow study.

### FuseMax-era and after

**FuseMax** [Nayak 2024]. *MICRO 2024*. Authors: Nandeeka Nayak, Xinrui Wu, Toluwanimi Odemuyiwa, Michael Pellauer, Joel S. Emer, Christopher Fletcher (UIUC/NVIDIA). **Architecture:** Cascade-of-Einsums taxonomy → spatial array with deep fusion + fine-grain pipelining; 100% compute utilization, no off-chip BW bottleneck, on-chip buffer independent of sequence length. 6.7× over FLAT iso-area. **Axes:** Dense, **FP16 fixed throughout** (no precision routing), prefill, fully spatial. **Direct precedent for our work** — our pitch extends FuseMax along three orthogonal axes: precision routing per attention stage (FuseMax pins FP16), compiler co-design (FuseMax uses Einsum spec only, no code generation), and RVV vector substrate (FuseMax uses a 256×256 spatial PE mesh). The IEEE Micro 45(4) 2025 retrospective ([Nayak 2025], DOI 10.1109/MM.2025.[11082638]) reframes the methodology but adds **zero new precision configs or compiler integration** — the mixed-precision dimension is genuinely unexplored. Workloads: BERT/T5/XLM/TrXL (encoder), **not** decoder-only LLMs with KV-cache growth.

**SystolicAttention / FSA** *arXiv:2507.11331* (2025). Closed-system systolic FA — full FA loop inside a single TPU-style systolic array. Closest closed-source competitor to FuseMax; no precision routing, no RVV.

**VMXDOTP** [Wipfli 2026]. *arXiv:2603.04979, DATE 2026*. Authors: Wipfli, Islamoglu, Benini (ETH Zurich PULP / same group as VEXP). **Architecture:** RVV custom dot-product instructions for MXFP8/MXFP4 on the **Spatz** RVV vector unit, 12nm. Evaluated on 64×64 DeiT-Tiny GEMM kernel. **Closest concurrent work** for our pitch. Differentiates from us on: (a) MX format (block 32, E8M0 scale) vs NVFP4 (block 16, E4M3 scale + per-tensor FP32 second-level scale); (b) unfused GEMM only, **no fused FA**; (c) hand-tuned LLVM intrinsics, **no compiler co-design**; (d) Spatz vector unit (in-order) vs Saturn (OoO + chaining + scoreboard); (e) image classifier (DeiT) vs decoder LLM (Llama-3.2-1B). Watch-list: Benini group ships RVV-customs-for-ML papers regularly (VEXP, VMXDOTP); plausibly an attention-flavored sequel for HPCA/MICRO 2027.

**Saturn microarch** *arXiv:2412.00997* [Zhao 2024]. Already in survey — the substrate.

### Quantization-algorithm precedents (algorithm side, no HW)

**ARCQuant** [Anonymous 2026]. *arXiv:2601.07475* (2026). NVFP4 post-training quantization algorithm for LLMs. Calibration + per-block scale assignment for NVFP4 K/V caches with <0.5 perplexity degradation reported on Llama-3.2. **Software only, no hardware.** Use as the quant-algorithm baseline that motivates our HW work: ARCQuant shows NVFP4 is accuracy-viable for K/V; we provide the HW + compiler stack to actually realize the bandwidth savings on RVV.

**MR-GPTQ "Bridging the Gap"** [arXiv:2509.23202, 2025]. MXFP4 quantization algorithm. Cited as part of the broader 4-bit quant trajectory; our work uses NVFP4 (NVIDIA-spec, different scaling) rather than MXFP4 (OCP-spec).

### FuseMax-enabling methodology papers (cite for the Einsum / fused-dataflow lineage)

**TeAAL** [Nayak 2024-ASPLOS]. *ASPLOS 2024*. Declarative sparse-tensor framework. Parent of FuseMax's Einsum-based dataflow methodology. We cite to contrast: **TeAAL is descriptive (specifies what the dataflow is), not generative (no code emitted)** — Exo's `@instr` system is generative.

**LoopTree** [Odemuyiwa 2024]. Fused-layer dataflow design space; same UIUC group. Cited by FuseMax for fusion analysis. Relevant for our fused-FA scheduling motivation.

**TileFlow** [Wu 2023]. *MICRO 2023*. Fusion-dataflow modeling tool used by FuseMax as a tool comparator. Cite to position our Exo-generated approach as code-generation rather than modeling.

**Choi attention accelerator** [Choi 2022-2023]. Alternate softmax-dataflow baseline that FuseMax compares to. Cited for the design-space context.

**Rabe & Staats** [Rabe 2022]. *arXiv:2112.05682*. "Self-attention does not need O(n²) memory" — the online-softmax precursor that both FuseMax and FlashAttention ground their formulations in. Cite as the algorithmic root.

### KV-cache and serving-tier accelerators (post-FuseMax)

**Oaken** *arXiv:2503.18599*. KV-cache-quantization accelerator.
**Kelle** *arXiv:2510.16040*. KV-cache RTL.
**Virgo** ASPLOS 2025. KV-cache-aware scheduling.
**KVQuant** NeurIPS 2024 (Shao).

(All in the running survey already — listed here only for the comparison table axes.)

### Industry + open-source software stack baselines

**NVIDIA Hopper / Blackwell Transformer Engine**. WGMMA + TMA + FP8 dynamic scaling; the de facto baseline. Hopper whitepaper public, microarch details not.

**Tenstorrent Wormhole / Blackhole.** Tensix tiles with 5 "Baby RISC-V" cores + SFPU (vector) + FPU (matrix); Tensix Vector is *disjoint* from RVV — proprietary ISA. Blackhole adds signed-shift support. The high-perf companion CPU **Tenstorrent Ascalon** is RVA23-compliant 8-wide OoO with 2×256-bit RVV units, but standard RVV — Tenstorrent's "custom" instructions live on the AI side, not the CPU.

**Groq LPU.** Deterministic systolic dataflow; attention is just another matmul + softmax pipeline stage. No public RTL.

**Esperanto ET-SoC-1.** 1088 ET-Minion in-order RISC-V cores each with proprietary vector/tensor extensions (256-bit FP / 512-bit INT8). ML-recommendation focus, not LLM.

**Cygnus** [Jain 2026]. *JSSC March 2026*. Octa-core RVV processor from Asanović/Shao/Nikolić (Berkeley). DSP-focused, no attention or quant work. Cite as recent Berkeley RVV-SoC context — shows Saturn's authors are publishing RVV silicon, but no LLM attention angle yet.

**llama.cpp upstream RVV path** (commit hashes per submission, ~2026 Q2). Two relevant kernels: (1) `ggml_v_expf_m2` in `ggml/src/ggml-cpu/vec.h:1342`, a polynomial RVV expf (~12 vector ops per iteration via range-reduce + minimax), used by `ggml_vec_soft_max_f32` (`vec.cpp:600-608`) and by the generic flash-attention path's inner softmax loop; (2) `forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16` and `_tiled_vlen1024_vf16` in `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1121, 1305`, **hardcoded VLEN=1024** fused FA for SpacemiT K1 only. **Direct upstream baselines for our work**: we beat the generic poly-RVV-softmax path on bandwidth (NVFP4 K/V) and beat SpacemiT's VLEN=1024-only path on portability (VLEN-parameterized via Exo). Cite both with commit hashes.

**SGLang RVV backend** [sgl-project/sglang #18072]. Software-only RVV 1.0 serving backend; FP16/INT8 paths, no hardware extension, no Exo. Cite as the production-serving-software comparison.

**IREE-RVV microkernel** [arXiv:2508.14899, Aug 2025]. SiFive's MLIR-RVV path on Intelligence X-series. Closed source SKL kernels behind vendor wall. Cite as the closed-source compiler precedent we differentiate from on open-source + custom-instruction awareness.

### Comparison table — open quadrant (rewritten 2026-05-16 PM for merged X+Y)

| Work | ISA target | Sparse? | Mixed prec? | Decode? | Fused FA? | Compiler co-design | OSS RTL |
|------|-----------|---------|-------------|---------|-----------|--------------------|---------|
| A³, ELSA | Fixed-function | yes (approx) | no (FP) | full | no | no | no |
| SpAtten | Fixed-function | yes (token/head) | partial (mixed-INT) | prefill | no | no | partial |
| Sanger / DOTA / Energon | Fixed + reconfig | yes (dyn) | partial | prefill | no | no | partial |
| FACT (ISCA'23) | Fixed-function | yes (eager-pred) | partial (mixed) | full TX | no | partial | no |
| FuseMax [MICRO'24] | Spatial array | no | **no (FP16 fixed)** | prefill | **yes** | Einsum spec only | sim only |
| SystolicAttention | Systolic | no | no | prefill | **yes** | none | no |
| Titopoulos [J.Supercomp'26] | RVV (gem5) | no | no | prefill | algo-level | none | sim only |
| VEXP [ARITH'25] | Snitch (Xfrep) | no | no (BF16) | softmax-only | no | LLVM intrinsic | partial |
| VMXDOTP [DATE'26] | Spatz (RVV) | no | **partial (MX, GEMM only)** | no (DeiT) | no | hand intrinsics | partial |
| SpacemiT FA [llama.cpp upstream] | K1 (VLEN=1024 only) | no | no (FP16) | yes (Llama-3.2) | yes (hand-tuned) | hand-tuned C++ | partial (kernel) |
| IREE-RVV [arXiv:2508.14899] | SiFive Intelligence | no | partial (INT8) | yes (Llama-3.2) | no | MLIR (closed) | no (SKL closed) |
| **This work** | **RVV 1.0 (Saturn)** | **no (v1)** | **yes (per-stage routing)** | **prefill+decode** | **yes (RTL)** | **Exo @instr DSL** | **yes (full stack)** |

**Open quadrant the applicant occupies:** *Open-source RVV-1.0 RTL with per-stage mixed-precision routing in a fused FA kernel, four custom precision-conversion + vfexp FUs, an Exo precision-routing scheduling pass, and end-to-end LLM via llama.cpp.* No prior work hits all six axes. The closest is VMXDOTP (custom RVV FU + MX precision, but no FA, no compiler, no LLM, in-order Spatz not Saturn) and SpacemiT's K1 FA (fused, LLM-ready, but FP16-only, VLEN=1024-only, vendor-closed RTL, hand-tuned).

---

## Gap 3 — RVV Custom-Instruction / FU-Extension Ecosystem

The space splits into four extension philosophies:

### Philosophy A: Custom RVV opcodes inside the vector pipeline ("VCIX-style")

**SiFive Intelligence X280 (and gen-2 X160/X180/X280/X390/XM).** Vector Coprocessor Interface eXtension (**VCIX**) — a custom accelerator hangs off the vector pipeline, executing custom *vector* opcodes using the standard RVV register file. SiFive Intelligence Extensions add ML-tuned ops; the X280 Gen-2 implements RVV 1.0 + Zvfh + the matrix extension (VME). **Docs:** sifive.com/blog/introducing-the-latest-sifive-intelligence-x280-processor, and the X280 manual on sifive.com/documentation.

**Andes AndesCore 45-series + ACE-RVV.** Andes Custom Extension (ACE) is a *full toolchain* for custom instructions: designer writes an ACE script (instruction semantics) + concise Verilog (RTL); the COPILOT tool generates intrinsics, simulator support, compiler patches. ACE-RVV variant targets the AX45MPV with up to 1024-bit VPU. **Docs:** andestech.com news 2023-03-23, COPILOT v6 docs (vendor-gated).

### Philosophy B: Custom vector opcodes via RVV custom-major-opcode space

**XuanTie C910 / C920 / C908 (Alibaba T-Head).** C910 implements RVV 0.7.1 (pre-ratification) with the **XTheadV** custom extension set; C920 brings RVV up to 1.0 but otherwise unchanged from C910. C908 (AIoT) supports RVV 1.0 + bitmanip + Zvfh + BF16 ops + XuanTie XMAE memory attribute extension + the XIE (XuanTie Instruction Extension) for proprietary instructions. **Docs:** github.com/XUANTIE-RV/openc910 (open-sourced C910 RTL), riscv.org/blog/xuantie-c908-high-performance-risc-v-processor-catered-to-aiot-industry. **Note for the applicant:** XTheadV is the *first* attempt at a vendor extension on a shipping core; useful precedent.

**SiFive Intelligence Extensions.** Sits inside the standard RVV major opcode but uses custom funct6 — distinct from VCIX, executes natively in the X-core vector pipeline.

### Philosophy C: Decoupled FU outside the core ("RoCC-style")

**Rocket Custom Coprocessor (RoCC).** Berkeley Rocket-tile mechanism: up to 4 RoCC accelerators per tile; instructions hit a separate decoder, execute in a custom block, write back through a side-channel to the integer regfile or memory. **Docs:** github.com/seldridge/rocket-rocc-examples, "The RoCC Doc V2" (chipyard docs). Used historically by **Hwacha** (Yunsup Lee), Berkeley's pre-RVV vector-fetch accelerator, which lived off the RoCC interface as the `Xhwacha` extension — *predates and does not implement* the ratified RVV 1.0.

### Philosophy D: FPU-integrated functional-unit extensions ("Snitch-style")

**Snitch / Occamy (ETH PULP).** Tiny pseudo-dual-issue RISC-V core with three orthogonal custom-ISA extensions:
- **Xssr (Stream Semantic Registers):** dedicated FP registers behave as memory streams — loads/stores happen implicitly, eliding explicit `vle`/`vse`.
- **Xfrep (FP Repetition):** hardware-issued loop of an FPU sequence buffer; no SW loop overhead. Effectively decouples FP issue from scalar control.
- **Xdma:** scalar core operates the DMA via custom instructions.
Plus *Smallfloat* (Xfvec) for sub-FP32 ops and the **VEXP** extension (arXiv:2504.11227, Wang et al., ARITH 2025) — bfloat16 exponentiation as an FPU custom op, 1% area, 162× softmax latency reduction. Saturn-relevant analog: VEXP-style integration of `vfexp` directly into the Saturn FP execution lane is the cleanest path. **Docs:** pulp-platform.github.io/snitch/rm/custom_instructions/.

### Philosophy E: Wholly disjoint ISA on a coupled tile

**Tenstorrent Wormhole / Blackhole.** Each Tensix has 5 baby RISC-V cores driving SFPU (vector) and FPU (matrix), where SFPU/FPU instructions are *not* RVV — they are Tensix-Vector, decoded separately. The RISC-V cores are "data-movers" / control. **Docs:** corsix.org/content/tt-wh-part5 (T-tile teardown), tt-wh-part6 (vector ISA).

**Esperanto ET-SoC-1.** ET-Minion cores have proprietary vector/tensor extensions on top of RV64. Not publicly documented at the encoding level.

### Saturn-specific approach

Per the running survey memory, Saturn's extension surface is the **FunctionalUnitFactory + decoder hook** pattern in the Chisel generator — a fresh FU is registered, the RVV decoder is extended for a new vector opcode in the custom-major-opcode region (or the funct6 funct3 reserved bits in the OP-V major opcode), and the FU participates in the issue/dispatch logic of the Saturn pipeline. This sits between Philosophy A (VCIX — vector-pipeline-internal but external module) and Philosophy D (Snitch — FPU-internal). **Docs:** saturn-vectors.org (manual) + github.com/ucb-bar/saturn-vectors. **Note:** I could not retrieve the manual page (10MB+ exceeded WebFetch limit); the FunctionalUnitFactory specifics are from the project memory and should be verified against the current `saturn-vectors` source when writing the camera-ready.

**Honesty flag — items I could not verify:**
- "BLOOM (Imperial)" as a custom RVV instruction framework: no hits. Possibly mis-remembered project name; should be searched against Imperial College London EEE/EIE publications (CAS group) directly.
- A Vivienne-Sze-authored "FACT" at HPCA 2023: does not exist in the search corpus; the FACT paper is Qin et al., ISCA 2023 (CAS/ICT). The Sze-adjacent attention/dataflow paper from this period is FLAT (Kao et al., ASPLOS 2023). The applicant should drop the "Sze 2023" attribution.
- LeapAttention (2023): not retrievable. Possibly confused with a different attention paper or never published broadly. Recommend dropping unless the applicant has a specific source.

### Critical positioning

Saturn's approach is **closer to VCIX (Philosophy A) than to RoCC (Philosophy C)**: the custom FU executes inside the vector pipeline with full access to the vector register file and the VL/vtype state, rather than being a decoupled coprocessor with its own scoreboard. It differs from Snitch (Philosophy D) in that Snitch's FREP-based scheme is *FPU-scalar* with stream registers replacing the vector regfile, whereas Saturn is a true RVV 1.0 vector unit. This is a *legitimate fifth point* in the design space, and the paper should explicitly position it as such — not claim novelty over RoCC or VCIX but argue it's the right one for the **compiler-co-design** story because Exo's `@instr` needs predictable cycle-accurate latencies inside the vector pipeline, not the asynchronous handoff of a RoCC interface.

---

## Output 4 — 1-Page Intro Outline (rewritten 2026-05-16 PM for merged X+Y)

See `intro_draft.txt` for the worked-out abstract/hook/contributions with risk flags. This is the condensed outline for fast review.

### Hook (1 paragraph)
Edge LLM inference has shifted from research to production: Apple Intelligence, Llama-3.2-1B/3B, Phi-3.5 mini, and Gemini Nano run on consumer devices. RISC-V is the emerging open-hardware substrate — SiFive Intelligence X-series, SpacemiT K1, Tenstorrent Ascalon, and Andes AX65 ship RVV 1.0 cores today. The open software stack has matured: upstream llama.cpp ships a polynomial RVV softmax (`ggml_v_expf_m2`) and a SpacemiT-tuned VLEN=1024 fused flash-attention path. But edge-LLM decode is *memory-bandwidth-bound* — each generated token reads the entire KV cache — and the open RVV stack pins all attention stages to one precision, leaving the headline KV-cache bandwidth savings (3.2× from 4-bit microscale formats like NVFP4 [ARCQuant 2026]) unrealized.

### Problem (1 paragraph)
The fundamental gap is not a missing op (softmax is vectorized in upstream) or a missing fused kernel (SpacemiT shipped one for K1) — it is the *co-design loop* between custom precision-conversion hardware and a compiler that decides safely which precision per attention stage. FuseMax [MICRO'24] pins FP16 throughout. VMXDOTP [DATE'26] adds MX RVV dot product on Spatz but unfused GEMM only, hand-tuned, image-classifier workload. VEXP [ARITH'25] adds a single softmax-exp instruction on Snitch's packed-FPU pipeline. SpacemiT's fused FA path is vendor-tuned to VLEN=1024, hand-coded C++, FP16-only. IREE-RVV [arXiv 2508.14899] is closed-source and lacks custom-instruction awareness. There is no open implementation combining (a) per-stage mixed-precision routing within a fused FA kernel, (b) custom RVV instructions that expose precision-conversion lanes to the compiler, (c) a scheduling DSL that routes precisions per stage, and (d) end-to-end deployment through llama.cpp.

### Approach (1 paragraph)
We close this loop on the **Saturn RVV 1.0 vector unit** [arXiv 2412.00997]. Four contributions stack: (1) four new RVV custom instructions — `vfconv.nvfp4.bf16.v` (3-cycle pipelined NVFP4 dequant with block-16 E4M3 scale), `vfconv.bf16.fp8.v` (2-cycle quant), `vfconv.fp8.bf16.v` (1-cycle dequant), and `vfexp.v` (8-cycle pipelined BF16→FP32 exp via range-reduce + degree-5 minimax) — implemented as Saturn FunctionalUnitFactory modules; ~0.14 mm² @ 16nm, ~3% of Saturn baseline; (2) a precision-routing scheduling pass in **Exo 2** [Ikarashi ASPLOS'25] that emits VLEN-parameterized fused flash-attention kernels with per-stage dtype: BF16 logits / NVFP4 K-V cache / FP32 softmax sum-accumulator / FP8-E4M3 attention weights / BF16 output; (3) Saturn integration of the four FUs + Exo-generated fused FA kernel running on Chipyard FireSim; (4) end-to-end deployment in llama.cpp via a new `GGML_RVV_NVFP4` backend variant, compared against upstream generic RVV FA and SpacemiT K1's hand-tuned VLEN=1024 FA. The differentiator from VMXDOTP, FuseMax, SpacemiT, and IREE-RVV is **per-stage precision routing via compiler-scheduled custom conversion lanes**: hardware exposes the dequant/quant lanes attention needs; the compiler decides safely which precision per stage; neither alone can realize the bandwidth savings.

### Contributions (bullet list)
- **Per-stage precision-routing analysis** for fused attention on an open vector ISA. First per-stage dtype-tolerance characterization (logits / softmax accum / KV cache / attn weights / output) on Llama-3.2-1B.
- **Four custom RVV instructions** on Saturn (`vfconv.nvfp4.bf16.v`, `vfconv.bf16.fp8.v`, `vfconv.fp8.bf16.v`, `vfexp.v`) implementing the conversion lanes and exp acceleration the precision-routed kernel needs.
- **Exo 2 precision-routing pass** with first-class `@instr` declarations for the four new instructions; emits VLEN-parameterized fused FA kernels.
- **Fused FA RTL** on Saturn integrating all four FUs; VLEN=128/256/512 demonstrated; bit-equivalent with hand-coded reference; throughput target ≥0.9× hand-coded.
- **End-to-end Llama-3.2-1B** via llama.cpp with a `GGML_RVV_NVFP4` backend; ≥1.5× decode speedup at ≤1.0 perplexity degradation vs SpacemiT K1 FP16 FA (Mo 10 checkpoint).
- **Open-source artifact**: Saturn RTL fork, Chisel testbenches, Exo `@instr` library, scheduled kernels, llama.cpp patch series, NVFP4 calibration data. Upstream PRs to Saturn / Exo / llama.cpp.

### Comparison Snapshot Table (mirrors `intro_draft.txt` Table 1)

| Axis | FuseMax [MICRO'24] | VEXP [ARITH'25] | VMXDOTP [DATE'26] | SpacemiT FA [llama.cpp upstream] | IREE-RVV [arXiv'25] | **This work** |
|------|-------------------|-----------------|-------------------|----------------------------------|---------------------|---------------|
| ISA / platform | Custom spatial | Snitch (Xfrep+SmallFloat) | Spatz (RVV) | K1 (VLEN=1024 only) | SiFive Intelligence | **RVV 1.0 (Saturn RTL)** |
| Open RTL | partial (model) | yes | yes | no (vendor) | no (vendor) | **yes** |
| Custom FU | full-attention FU | softmax-exp | MX dot product | none | none | **4 (NVFP4↔BF16↔FP8 + vfexp)** |
| Mixed prec routing | no (FP16) | no (BF16) | partial (MX, GEMM only) | no (FP16) | partial (INT8) | **yes (per-stage)** |
| Compiler co-design | Einsum spec only | LLVM intrinsics | hand intrinsics | hand-tuned C++ | MLIR (closed) | **Exo 2 @instr DSL** |
| Fused FA | yes | no (softmax only) | no | yes (VLEN=1024 only) | no | **yes (VLEN-parametric)** |
| End-to-end LLM | no (BERT/T5) | no (FA microbench) | no (DeiT image) | yes (Llama-3.2) | yes (Llama-3.2) | **yes (Llama-3.2-1B)** |
| OSS stack | no | partial | partial | partial (kernel only) | no | **yes (full stack)** |

---

## References (consolidated; URL/arXiv ID + venue)

1. Dao et al. **FlashAttention.** arXiv:2205.14135 — NeurIPS 2022. https://arxiv.org/abs/2205.14135
2. Dao. **FlashAttention-2.** arXiv:2307.08691 — ICLR 2024. https://arxiv.org/abs/2307.08691
3. Shah, Bikshandi, Zhang et al. **FlashAttention-3.** arXiv:2407.08608 — NeurIPS 2024. https://arxiv.org/abs/2407.08608
4. Dao, Haziza, Massa, Sizov. **Flash-Decoding for long-context inference.** Stanford CRFM blog, 2023-10-12. https://crfm.stanford.edu/2023/10/12/flashdecoding.html
5. Hong, Dai, Xu et al. **FlashDecoding++.** arXiv:2311.01282 — MLSys 2024. https://arxiv.org/abs/2311.01282
6. Ye, Chen, Lai et al. **FlashInfer.** arXiv:2501.01005 — MLSys 2025 (Best Paper). https://arxiv.org/abs/2501.01005
7. Zhang et al. **SageAttention.** arXiv:2410.02367 — ICLR 2025. https://arxiv.org/abs/2410.02367
8. Zhang et al. **SageAttention2.** arXiv:2411.10958. https://arxiv.org/abs/2411.10958
9. Zhang et al. **SageAttention3 (MXFP4).** arXiv:2505.11594. https://arxiv.org/abs/2505.11594
10. Liu, Zaharia, Abbeel. **Ring Attention.** arXiv:2310.01889. https://arxiv.org/abs/2310.01889
11. Sun et al. **BurstAttention.** arXiv:2403.09347. https://arxiv.org/abs/2403.09347
12. Brandon et al. **Striped Attention.** arXiv:2311.09431. https://arxiv.org/abs/2311.09431
13. Shazeer. **Fast Transformer Decoding (MQA).** arXiv:1911.02150.
14. Ainslie et al. **GQA.** arXiv:2305.13245 — EMNLP 2023. https://arxiv.org/abs/2305.13245
15. DeepSeek-AI. **DeepSeek-V2 (MLA).** arXiv:2405.04434. https://arxiv.org/abs/2405.04434
16. Lin et al. **Hardware-Centric Analysis of MLA.** arXiv:2506.02523. https://arxiv.org/abs/2506.02523
17. Ham, Jung, Kim et al. **A³.** HPCA 2020. https://taejunham.github.io/data/a3_hpca2020.pdf
18. Ham, Lee, Kim et al. **ELSA.** ISCA 2021. https://taejunham.github.io/data/elsa_isca21.pdf
19. Wang, Zhang, Han. **SpAtten.** HPCA 2021 — arXiv:2012.09852. https://hanlab.mit.edu/projects/spatten
20. Lu, Jin et al. **Sanger.** MICRO 2021. doi:10.1145/3466752.3480125
21. Qu, Liu, Tu et al. **DOTA.** ASPLOS 2022. doi:10.1145/3503222.3507738
22. Zhou et al. **Energon.** HPCA 2022 / IEEE TC.
23. Qin et al. **FACT.** ISCA 2023. doi:10.1145/3579371.3589057
24. Wang, Xu, Wang, Han. **CTA.** HPCA 2023. doi:10.1109/HPCA56546.2023.10070997
25. Kao, Subramanian et al. **FLAT.** ASPLOS 2023.
26. Nayak, Wu, Odemuyiwa, Pellauer, Emer, Fletcher. **FuseMax.** MICRO 2024 — arXiv:2406.10491. https://arxiv.org/abs/2406.10491
27. SystolicAttention / FSA. arXiv:2507.11331.
28. Titopoulos et al. **gem5 RVV FA.** arXiv:2510.06834.
29. Wang, Islamoglu, Belano et al. **VEXP.** arXiv:2504.11227 — ARITH 2025. https://arxiv.org/abs/2504.11227
30. Zhao et al. **Saturn microarchitecture / Instruction Scheduling.** arXiv:2412.00997. https://saturn-vectors.org/
31. Ikarashi et al. **Exo 2.** arXiv:2411.07211 — ASPLOS 2025.
32. SiFive. **Intelligence X280 + VCIX.** https://www.sifive.com/blog/introducing-the-latest-sifive-intelligence-x280-processor
33. SiFive. **Intelligence Family Gen-2 (X160/180/280/390/XM).** Press 2025-09-08. https://www.sifive.com/press/new-x100-series-second-gen-intelligence-family
34. Andes Technology. **ACE + AndesCore 45-series.** Press 2023-03-23. http://www.andestech.com/en/2023/03/23/andes-custom-extension-ace-supports-andescore-45-series-processors-to-provide-flexible-acceleration/
35. Alibaba T-Head. **OpenC910 (RVV 0.7.1).** https://github.com/XUANTIE-RV/openc910
36. Alibaba T-Head. **XuanTie C908 RISC-V (RVV 1.0).** RISC-V Blog 2022. https://riscv.org/blog/xuantie-c908-high-performance-risc-v-processor-catered-to-aiot-industry-chang-liu-alibaba-cloud/
37. Tenstorrent. **Wormhole Series Part 6: Vector ISA.** corsix.org/content/tt-wh-part6.
38. Jones (Tenstorrent). **Ascalon RISC-V CPU.** RISC-V Summit 2025. https://tenstorrent.com/en/ip/risc-v-cpu
39. Ditzel et al. **ET-SoC-1.** IEEE Hot Chips. https://www.esperanto.ai/wp-content/uploads/2022/05/Dave-IEEE-Micro.pdf
40. PULP. **Snitch custom instructions (Xssr, Xfrep, Xdma).** https://pulp-platform.github.io/snitch/rm/custom_instructions/
41. Zaruba et al. **Snitch: tiny pseudo dual-issue.** https://htor.ethz.ch/publications/img/zaruba-snitch.pdf
42. Asanović et al. **Rocket Chip Generator (RoCC).** UCB-EECS-2016-17. https://digitalassets.lib.berkeley.edu/techreports/ucb/text/EECS-2016-17.pdf
43. Lee, Ou et al. **Hwacha.** UCB-EECS-2015-262. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-262.pdf
44. ggml. **RVV 128-bit support (PR merged 2025-03-27).** https://github.com/ggml-org/llama.cpp
45. PLCT Lab. **llama.cpp RVV PR announcement.** https://plctlab.org/en/news/085/
46. Cloud-V. **Accelerating llama.cpp with RVV.** https://cloud-v.co/blog/risc-v-1/accelerating-llama-cpp-with-risc-v-vector-extension-3
47. SiFive. **LLM optimization on Intelligence (TinyLlama, Llama2-7B-Q4).** https://www.sifive.com/blog/llm-optimization-and-deployment-on-sifive-intellig

### Additions 2026-05-16 PM (merged X+Y rewrite)

48. Wipfli, Islamoglu, Benini. **VMXDOTP: MX dot-product instructions for RVV on Spatz.** arXiv:2603.04979 — DATE 2026. https://arxiv.org/abs/2603.04979
49. (anonymous). **ARCQuant: NVFP4 post-training quantization for LLMs.** arXiv:2601.07475. https://arxiv.org/html/2601.07475v1
50. Jain, Asanović, Shao, Nikolić. **Cygnus: octa-core RVV processor.** IEEE JSSC, March 2026.
51. SGLang RVV serving backend. https://github.com/sgl-project/sglang/issues/18072
52. Nayak, Pellauer, Emer, Fletcher et al. **From TeAAL to FuseMax: Separation of Concerns for Attention Accelerator Design.** IEEE Micro 45(4) 2025 — DOI 10.1109/MM.2025.[11082638]. https://ieeexplore.ieee.org/document/11082638/
53. Nayak et al. **TeAAL: a declarative sparse-tensor framework.** ASPLOS 2024. (parent of FuseMax's Einsum methodology)
54. Odemuyiwa et al. **LoopTree: fused-layer dataflow design space.** TACO / PACT 2024.
55. Wu et al. **TileFlow: fusion-dataflow modeling.** MICRO 2023.
56. Rabe, Staats. **Self-attention does not need O(n²) memory.** arXiv:2112.05682, 2022. (online-softmax precursor)
57. (Anonymous Choi et al.) attention dataflow baseline. ~2022-2023 — verify exact citation from FuseMax bibliography.
58. **llama.cpp upstream RVV softmax** (`ggml_v_expf_m2`, `ggml/src/ggml-cpu/vec.h:1342`) — cite commit hash + PR# at submission time.
59. **SpacemiT VLEN=1024 fused FA in llama.cpp** (`forward_flash_attn_ext_f16_{one_chunk,tiled}_vlen1024_vf16`, `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1121, 1305`) — cite commit hash + PR# at submission time.

---

## Surprising findings + adversarial review prep

(Internal note for the applicant, not for the paper.)

**Updated 2026-05-16 PM after merged X+Y rewrite.** Six findings worth tracking:

1. **FlashDecoding is a blog post, not a paper.** Tri Dao never wrote it up formally. A reviewer will ask "your decode story is built on a non-peer-reviewed blog reference"; cite FlashInfer (MLSys'25) or FlashDecoding++ (Hong et al. MLSys'24) instead.

2. **"FACT @ HPCA 2023 by Vivienne Sze" does not exist.** Sze's group has FLAT (ASPLOS'23, dataflow analysis, not a chip); the FACT paper is Qin et al. at ISCA'23 from CAS/ICT. Drop the Sze attribution.

3. **Tenstorrent's "RVV custom ops" framing is misleading.** Ascalon is plain RVA23 RVV; Wormhole/Blackhole use Tensix-Vector which is *not* RVV. Correct framing: "Tenstorrent ships standard RVV on Ascalon and a separate proprietary vector ISA on the AI side."

4. **SiFive's VCIX ≠ SiFive Intelligence Extensions.** VCIX is the external-accelerator interface; Intelligence Extensions are in-pipeline custom funct6s. Don't conflate.

5. **Upstream llama.cpp closed the "scalar RVV softmax" gap** in 2025-2026 via `ggml_v_expf_m2` polynomial expf in `vec.h:1342`, used by both the generic softmax kernel and the generic flash-attention path. The "fill the RVV softmax gap with vfexp" framing from an earlier draft is dead. SpacemiT also shipped a K1-tuned VLEN=1024 fused FA in `spacemit/rvv_kernels.cpp`. **New differentiation must lead with per-stage mixed-precision routing**, not "first RVV softmax." Anticipated adversarial question: *"You're competing against SpacemiT's already-fast fused FA — why does adding NVFP4 quant lanes matter?"* — answer: K1's fused path is FP16-only and VLEN=1024-hardcoded; we (a) cut KV-cache bandwidth 3.2× by routing precisions per stage, (b) generalize to any VLEN via compiler scheduling, (c) demonstrate the compiler co-design loop on real silicon.

6. **VMXDOTP (Benini, DATE 2026) is the closest concurrent work.** Differentiate explicitly: (a) NVFP4 (block 16, E4M3 scale) vs MX (block 32, E8M0 scale); (b) fused FA pipeline vs unfused GEMM kernel; (c) Exo-scheduled compiler vs hand-tuned LLVM intrinsics; (d) Saturn (OoO + chaining + scoreboard) vs Spatz (in-order); (e) Llama-3.2-1B decoder LLM with KV-cache growth vs DeiT image classifier. **Benini group continues to publish RVV-customs-for-ML papers (VEXP, VMXDOTP); plausibly an attention-themed sequel for HPCA/MICRO 2027** — set arXiv keyword alerts.

7. **Anticipated adversarial question on vfexp:** *"How is `vfexp` different from VEXP — what makes this a paper?"* — the answer must be that vfexp is one component of the precision-routing FU family, not the headline contribution. The compiler-DSL co-design with Exo (which schedules vfexp alongside the three conversion-lane instructions) is the paper. The FU alone is incremental over VEXP.

# Quantization-related HW alternatives for arch+compiler PhD signal

**Date:** 2026-05-16. Scope: ~12-month side-project; full-stack (HW+compiler+ASIC+SW) required. Comparison baseline = Saturn `vfexp` + fused FA RTL + MLIR FA-lowering pass + llama.cpp RVV.

---

## Direction 1: NVFP4-on-Gemmini as primary contribution

**(a) OSS state (May 2026).**
- RTL: `ucb-bar/gemmini` `gemmini-mx` branch is the only open MX-aware Gemmini tree. Last public push **Mar 28, 2026**; block size 32 / E8M0 scale hardcoded in `MxRequantizer.scala` (~lines 187–188, 366–376). No NVFP4 (block 16, E4M3 scale, FP32 second-level) anywhere in any open accelerator. Voyager (Stanford, arXiv:2509.15205) ships **MXINT8** only, no FP4. Vortex GPGPU has no microscaling support.
- Compiler: LLVM/MLIR has `APFloat` E2M1 (PR #95392) and E8M0 (PR #107127) merged. **No MX→RVV lowering, no NVFP4 type, no `mx` MLIR dialect upstream.** TPP-MLIR, IREE, and Triton each have ad-hoc MXFP4 paths for GPUs; none target RISC-V or Gemmini RoCC.
- SW: llama.cpp has 3 RVV files, **zero** FP4/MXFP4/NVFP4 kernels. vLLM has NVFP4 paths via Marlin/Machete for Blackwell only. SGLang and MLC have MXFP4 weight-only for CUDA.

**(b) Most-recent papers.** NVIDIA "Pretraining LLMs with NVFP4" arXiv:2509.25149 (Sep 2025); "Bridging the Gap" MR-GPTQ MXFP4 arXiv:2509.23202 (ICLR 2026); "Four Over Six" adaptive NVFP4 block scaling arXiv:2512.02010; RaZeR NVFP4 zero-remapping arXiv:2501.04052. **M2XFP** at ASPLOS 2026 (arXiv:2601.19213, SJTU) already proposes metadata-augmented MX with a hardware unit, but it is **not** NVFP4-specific and targets a separate accelerator, not Gemmini.

**(c) Verifiable gap.** No open-source RTL accelerator (Gemmini, Voyager, Vortex, NVDLA, Saturn) implements native NVFP4 dot-products with the two-level scale (block-E4M3 × tensor-FP32), and no MLIR pass lowers a quantized linear op to NVFP4 RoCC instructions.

**(d) Faculty alignment.** Strong.
- **Sophia Shao (Berkeley)** — owns Gemmini and KVQuant (NeurIPS 2024) and Virgo (ASPLOS 2025); NVFP4-on-Gemmini sits directly on her stack. Risk: also overlap-prone (see (g)).
- **Tushar Krishna (GT)** — MicroScopiQ (ISCA 2025, arXiv:2411.05282) and FlexiBit (arXiv:2411.18065) are exactly outlier-aware MX accelerator papers; he would read an NVFP4 RTL paper carefully.
- **Mark Horowitz (Stanford)** — Voyager (arXiv:2509.15205) is the MXINT8 competitor; an NVFP4 Gemmini extension is a natural companion piece. Brucek Khailany (NVIDIA/Stanford-adjacent) is the NVFP4 spec author.
- **Mohamed Abdelfattah (Cornell Tech)** — BitMoD (HPCA 2025) is mixed-datatype LLM acceleration; close fit.

**(e) Compiler co-design strength.** Moderate. Real work exists: (1) defining an MLIR `mx` dialect or extending the `quant` dialect with NVFP4 type ops; (2) writing a Linalg→Gemmini-RoCC lowering that respects 16-element blocks and the two-level scale; (3) a `tosa.rescale`-style pass that picks per-tensor FP32 scales offline. But this is mostly **type-system plumbing**, not algorithmic compiler research. The "co-design" pitch is real but thinner than baseline's FA-lowering pass.

**(f) Realistic 12-month scope.** Tight.
- RTL: rewriting `MxRequantizer.scala` and the MAC tiles for block-16 / E4M3 / FP32 second-level ≈ **1500–3000 LOC Chisel, 2–4 months** including verification against a software golden model.
- Compiler: MLIR `mx`/`quant` extension + Linalg→Gemmini-RoCC pass ≈ **2–3 months**.
- SW: NVFP4 dequant + GEMM in llama.cpp (or TinyChat-style runtime) ≈ **1–2 months**.
- Eval: end-to-end Llama-3-8B perplexity + cycle-accurate Spike/FireSim ≈ **2 months**.
**Total realistic = 8–11 months for a workshop paper; tight but doable.** No vector unit work; no fused softmax. The compiler story is the weakest leg.

**(g) Risks.** **High.**
- Hansung Kim / Rakela / Amanda Shi are actively pushing `gemmini-mx`; NVFP4 is the obvious next branch commit. Direct overlap risk with the **owning lab**.
- M2XFP (ASPLOS 2026) already covers metadata-augmented MX; reviewers will ask "vs. M2XFP?" — answerable but adds writing burden.
- NVIDIA could open-source NVFP4 dequant kernels for an open accel (low probability but non-zero).
- ISCA/MICRO 2026 likely has 2–3 NVFP4 hardware papers from industry teams; novelty window narrows fast.

**(h) Comparison to baseline.** **Slightly weaker as a PhD-application signal**, despite being more "topical." Reasons: (1) Shao-lab overlap is a real concern when applying to Berkeley; (2) compiler story is type-plumbing, not algorithmic — your advisor cares about *compiler co-design* and FA-lowering is a richer compiler problem; (3) NVFP4 is highly competitive and your work risks being scooped by an industry preprint mid-submission. Baseline differentiates better.

---

## Direction 2: Mixed-precision attention RTL (FP16 logits / FP4 KV / FP8 softmax accum)

**(a) OSS state.** No open RTL accelerator implements heterogeneous-precision attention. Saturn supports per-instruction `vtype` precision in software but no fused FA path. Gemmini RoCC accelerates GEMM, not softmax. SageAttention3 (arXiv:2505.11594, NeurIPS 2025 Spotlight) implements FP4-microscaling attention on **Blackwell only**, software kernel. vLLM FP8 KV-cache (April 2026 blog) covers FP8 K/V + FP8 QK/PV but is GPU CUDA. FireQ (arXiv:2505.20839) is INT4-FP8 attention on Hopper. KVTuner (arXiv:2502.04420), MixKVQ (arXiv:2512.19206), HACK (arXiv:2502.03589) are all software/algorithmic, no RTL.

**(b) Most-recent papers.** SageAttention3 (arXiv:2505.11594, May 2025); FireQ (arXiv:2505.20839); MixKVQ (arXiv:2512.19206); MX+ MICRO 2025 (arXiv:2510.14557 — outlier-aware microscaling for serving, *not* attention specifically); FuseMax (Emer, MICRO 2024) — closest attention-accelerator prior art but precision-uniform.

**(c) Verifiable gap.** No open RTL fused-attention block performs per-operand precision selection (Q/K/V/logits/softmax/output) with a compiler-managed precision-routing policy; SageAttention3 is closest in spirit but is a CUDA kernel only.

**(d) Faculty alignment.** **Strongest of the three.**
- **Joel Emer (MIT/NVIDIA)** — FuseMax (MICRO 2024) is precisely the einsum-attention-accelerator template; precision-routing extension reads as a direct follow-on.
- **Sophia Shao** — KVQuant is per-channel / per-token mixed-precision KV; HW realization is open.
- **Tushar Krishna** — FlexiBit explicitly targets arbitrary mixed precision; FlashAttention with FlexiBit-style PEs is on-brand.
- **Vivienne Sze (MIT)** — Eyeriss-line cares about per-layer precision selection.

**(e) Compiler co-design strength.** **Strongest.** A real compiler problem: a precision-routing pass that, given a calibration profile (KVQuant-style per-channel sensitivity), selects per-tensor precisions, inserts cast nodes, and verifies the softmax accumulator never overflows. This is a *policy* pass (cost model + search), not just type plumbing. Maps cleanly to MLIR `quant` dialect + a custom `attention` op.

**(f) Realistic scope.** **Tightest.**
- RTL: fused attention block on Saturn or as Gemmini extension with mixed-precision MAC tiles ≈ **4–6 months Chisel**.
- Compiler: precision-routing pass + cost model + calibration ingestion ≈ **3–4 months**.
- SW: llama.cpp RVV mixed-precision attention kernel ≈ **1–2 months**.
- Eval: 1–2 months. **Total = 9–13 months — at the edge of feasibility for a workshop paper.**

**(g) Risks.** **Moderate.**
- FuseMax follow-ons likely at MICRO/ISCA 2026 from Emer's group.
- SageAttention3 / FireQ keep raising the GPU bar; hard to claim end-to-end speedup vs. Blackwell.
- Less overlap with Shao group than D1 (Virgo is GPGPU, KVQuant is sw).

**(h) Comparison to baseline.** **Strictly stronger as PhD signal IF you can land the scope.** Reasons: (1) algorithmic compiler pass (precision-routing) is what an LLM-arch+compiler-co-design advisor most wants to see; (2) clean alignment with Emer/Shao/Krishna; (3) less head-on collision with the gemmini-mx maintainers' roadmap. **Risk: 1.3× scope over baseline. The vfexp+FA baseline already taught you the attention block and exp approximation; this is the natural escalation, not a side-step.**

---

## Direction 3: Dynamic activation quantization HW (SmoothQuant/AWQ-style at HW)

**(a) OSS state.** OPAL (DAC 2024, arXiv:2409.05902) and MicroScopiQ (ISCA 2025, arXiv:2411.05282) both implement outlier-preserved MX accelerators **in simulator, not Chipyard-grade open RTL**. OSC (arXiv:2604.12782) does W4A4 with FP8 fallback. M2XFP (ASPLOS 2026) has a lightweight HW unit but again not in an open Chipyard accelerator. **No Gemmini/Saturn open implementation. No MLIR pass that decides per-layer/per-token quant strategy.** vLLM's per-token dynamic FP8 is software-only.

**(b) Most-recent papers.** MicroScopiQ (ISCA 2025); OPAL (DAC 2024); M2XFP (ASPLOS 2026, arXiv:2601.19213); MX+ (MICRO 2025, arXiv:2510.14557); OSC (arXiv:2604.12782). The field is **the most crowded** of the three.

**(c) Verifiable gap.** No open RTL implementation of an outlier-aware scale-recomputation unit integrated into a Chipyard SoC, and no MLIR pass that emits the runtime scale-recomputation code.

**(d) Faculty alignment.** Moderate.
- **Krishna (GT)** — MicroScopiQ is exactly this; he is the strongest match but also **publishes here aggressively**, raising overlap risk.
- **Abdelfattah (Cornell Tech)** — BitMoD.
- **Shao** — KVQuant outlier handling is algorithmic; HW realization is open.

**(e) Compiler co-design strength.** **Moderate.** A pass that picks per-layer/per-token strategy is interesting, but most papers in this space treat the strategy as offline-calibrated, which dilutes the "compiler" novelty. Runtime-dispatched precision needs HW hooks that are themselves the contribution; the compiler ends up being a scheduling pass.

**(f) Realistic scope.** **Largest of the three.**
- RTL: outlier-detection unit + dual-precision PE + scale-recompute pipeline in Chipyard ≈ **4–7 months Chisel**, much higher than D1.
- Compiler: per-layer strategy selection pass + calibration ingestion ≈ **2–3 months**, but novelty is contested.
- SW: 1–2 months.
- Eval: 2 months. **Total = 9–14 months; you will compete directly with MicroScopiQ follow-ons.**

**(g) Risks.** **Highest.**
- MicroScopiQ, OPAL, MX+, M2XFP, OSC — **five** competing recent papers, three with HW. Workshop reviewers will ask "vs. each of these" and the answer must be defensible.
- Krishna's group likely has follow-ups in flight for ISCA/MICRO 2026.
- NVIDIA's NVFP4 with FP32 second-level scale partially obviates the outlier story.

**(h) Comparison to baseline.** **Weaker.** Crowded field, weakest compiler novelty, largest RTL scope. Only pick this if you have an specific outlier-handling insight that the existing five papers miss.

---

## Bottom-line ranking for arch+compiler-co-design PhD signal

1. **Baseline (Saturn `vfexp` + fused FA + MLIR FA-lowering + llama.cpp RVV)** — strongest compiler pass, unique RVV target, no direct lab overlap, scope realistic. **Keep as primary.**
2. **D2 (mixed-precision attention RTL)** — strongest *if* you can land scope; strongest faculty alignment (Emer, Shao, Krishna, Sze); compiler pass (precision routing) is algorithmic. **Consider as alternative or as v2 of baseline.**
3. **D1 (NVFP4-on-Gemmini)** — solid but overlap risk with Shao lab + thin compiler story. Good if you want a fast 6-month follow-on after baseline ships.
4. **D3 (dynamic activation quant HW)** — most crowded, weakest compiler novelty, largest RTL scope. **Avoid as primary.**

**Recommendation.** Keep baseline; if you want a quantization angle, fold a single NVFP4 KV-cache path into the FA RTL (mini-D1 as 1–2 month addendum, 200–400 LOC Chisel) rather than swapping. The baseline's `vfexp` + FA-lowering compiler pass is the rarest signal in this faculty list and dominant for an LLM-arch+compiler-co-design advisor.

---

### Cited URLs / arXiv IDs

- NVFP4 papers: arXiv:2509.25149, arXiv:2509.23202, arXiv:2512.02010, arXiv:2501.04052
- MicroScopiQ ISCA 2025: arXiv:2411.05282
- FlexiBit: arXiv:2411.18065
- OPAL DAC 2024: arXiv:2409.05902
- MX+ MICRO 2025: arXiv:2510.14557
- M2XFP ASPLOS 2026: arXiv:2601.19213, https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26
- Voyager (Stanford): arXiv:2509.15205, https://code.stanford.edu/voyager/accelerator
- Virgo (Shao/Kim, ASPLOS 2025): arXiv:2408.12073, https://github.com/ucb-bar/virgo
- KVQuant (Shao, NeurIPS 2024): https://slice.eecs.berkeley.edu/papers/kvquant-towards-10-million-context-length-llm-inference-with-kv-cache-quantization/
- Saturn: https://github.com/ucb-bar/saturn-vectors, EECS-2024-215
- Vectorized FA on RVV: arXiv:2510.06834; VEXP arXiv:2504.11227
- SageAttention3: arXiv:2505.11594
- FireQ: arXiv:2505.20839
- KVTuner/MixKVQ/HACK: arXiv:2502.04420, arXiv:2512.19206, arXiv:2502.03589
- OSC W4A4: arXiv:2604.12782
- vLLM FP8 KV-cache blog (Apr 2026): https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- Gemmini repo + `gemmini-mx` branch (last update Mar 28, 2026): https://github.com/ucb-bar/gemmini
- BitMoD (Abdelfattah HPCA 2025): https://github.com/abdelfattah-lab/BitMoD-HPCA-25
- FuseMax (Emer MICRO 2024): see https://people.csail.mit.edu/emer/publications/

# RISC-V side-project — deep survey synthesis (2026-05-16, **annotated 2026-05-16 PM after Y1 pivot**)

Preserved for reference. The Y1 direction lock (originally Proposal X = single-FU vfexp + Exo + FA + llama.cpp) was made downstream of this synthesis after the Fall 2030 timeline correction (43 months available, not 19).

**Pivot 2026-05-16 PM**: After upstream-state recheck found `ggml_v_expf_m2` polynomial RVV softmax + SpacemiT VLEN=1024 fused FA already in upstream llama.cpp, the original Proposal X premise died. Y1 was widened to **merged X+Y** (mixed-precision fused FA via 4 custom RVV instructions + Exo precision-routing pass). Old "Y1 LOCKED" tags below are HISTORICAL — current Y1 is merged X+Y per [[project-side-project-direction]].

## Cross-survey comparison

| Candidate | HW nov | Comp nov | 12-mo scope | Concurrent risk | Faculty fit | vs baseline |
|---|---|---|---|---|---|---|
| Baseline (Saturn vfexp + MLIR FA) | Med | Med | Low risk | Med (VEXP/Titopoulos) | Broad | — |
| A1. Block-sparse attention | Med-High | High | Med | Med-High (Han) | Han / Krishna / Abdelfattah | Stronger |
| A2. KV-cache RTL | High | Med | High | High (Kelle/Oaken/Tender) | Shao | Weaker for solo |
| A3. Sink / Streaming | Low-Med | Low | Low | Med | Han/Krishna | Add-on only |
| Q1. NVFP4-on-Gemmini | Med-High | Low-Med | Tight | High (Hansung Kim overlap) | Shao/Krishna/Horowitz | Weaker (Berkeley overlap) |
| Q2. Mixed-precision attention | High | High | Tight (9-13 mo) | Med | Emer / Shao / Krishna / Sze | Strictly stronger if scope holds |
| Q3. Dynamic activation quant | Med | Low-Med | High | High (5+ papers) | Krishna | Weaker |
| S1. MoE dispatch | High | High | Med | High (EARTH ASPLOS'26) | Krishna/Shao/Abdelfattah | Higher ceiling, weaker floor |
| S2. Spec decode on Saturn | Med | High | Low risk | Med | Han/Shao/Dao | Comparable; better narrative |
| S3. Prefill/decode disagg | High | Highest | Infeasible solo | Med | Wentzlaff/Patel/Shao/Kozyrakis | Headline > deliverable |
| C1. Triton-CPU-RVV | Low | Med | Med | High (Terapines, ML-Triton) | Weak | Weaker (no co-design cred) |
| C2. Exo-RVV + @instr vfexp | Med | High | Low risk | Med (Ragan-Kelley own lab) | Ragan-Kelley direct / Amarasinghe / Batten | Stronger for compiler-DSL advisor |
| C3. IREE-RVV | Low-Med | Med | High | High (SiFive ships it) | Reddi / Ceze (weak) | Weaker (novelty surface ~0) |

## Cross-cutting findings

1. VEXP (arXiv:2504.11227, Snitch, ARITH 2025) is the dominant concurrent-work hazard for any `vfexp`-style claim. Already shipped on a different RVV-adjacent core with 162x softmax / 8.2x FA-2 speedup. Differentiation must be RVV-1.0-native + Saturn-integrated + compiler-aware.
2. 3 of 4 best alternatives are still attention-centric. Attention is the right primary surface; question is flavor.
3. NVFP4-on-Gemmini as primary = direct overlap with Hansung Kim / gemmini-mx team at Berkeley. Better as 1-2 mo addendum.
4. Song Han's lab owns sparse/streaming attention. Beat on open RVV + RTL + MLIR end-to-end, not algorithmic novelty.
5. Compiler novelty quality > quantity. Exo @instr scheduling, mixed-precision routing, and sparse pattern matching all materially stronger than MLIR FA-lowering.
6. Kelle (MICRO'25) RTL is closed-source, foreclosing direct apples-to-apples comparison for KV-cache RTL direction.

## Three refactored proposals (updated 2026-05-16 PM after Y1 pivot to merged X+Y)

**X+Y MERGED — Mixed-precision fused FA on Saturn + 4 custom RVV instructions + Exo precision-routing + llama.cpp** [**Y1 LOCKED 2026-05-16 PM**]
Saturn FU with four new RVV instructions (`vfconv.nvfp4.bf16.v` / `vfconv.bf16.fp8.v` / `vfconv.fp8.bf16.v` / `vfexp.v`) + Exo 2 precision-routing pass + llama.cpp NVFP4 backend variant + Llama-3.2-1B E2E. Per-stage precision routing: BF16 logits / NVFP4 K-V cache (block 16, E4M3 scale) / FP32 softmax sum / FP8-E4M3 attention weights / BF16 output.
- Mixed-precision is the headline novelty (not vfexp). vfexp is one of four FUs.
- Direct FuseMax (MICRO 2024) follow-up on a new ISA, new compiler, new precision config. FuseMax pins FP16; we route per stage.
- VLEN-parameterized via Exo, generalizing beyond SpacemiT's K1 VLEN=1024-only path.
- 12 mo critical path.
- **Risk #1**: VMXDOTP (Benini DATE 2026) ships MX RVV dot on Spatz — differentiate on NVFP4 (not MX), fused FA (not unfused GEMM), Exo (not hand intrinsics), Saturn (not Spatz), LLM (not DeiT image).
- **Risk #2**: Ragan-Kelley group may publish Exo-RVV before our Mo 8. Exo added ARM SVE May 2025, NO RVV yet. Email Ragan-Kelley's group as collaboration heads-up by Mo 4.
- **Risk #3**: NVFP4 quantization accuracy on Llama-3.2-1B — needs ARCQuant tooling validated by Mo 6 before Mo 10 perplexity claims.

**~~X (original, single-FU vfexp + Exo + FA + llama.cpp)~~** [SUPERSEDED 2026-05-16 PM]
Original Proposal X has been absorbed into the merged X+Y direction. vfexp is preserved as one of the four custom instructions; the rest of the X stack (Exo + FA + llama.cpp) is preserved as components of the merged Y1. **Do not pitch as standalone** — upstream `ggml_v_expf_m2` closed the "fill the scalar softmax gap" framing.

**~~Y (mixed-prec FA, originally Y2 candidate)~~** [PROMOTED TO Y1 2026-05-16 PM]
Originally planned as Y2 extension of the locked X. After the upstream-state recheck, the mixed-precision angle became the strongest available headline, so promoted to Y1. The Y2 slot is now filled by **scaling-up** the merged X+Y (larger model / longer context / per-layer hetero precision).

**Z — Swap algorithm to block-sparse Mistral** [Y3 ORTHOGONAL CANDIDATE, unchanged]
Saturn + block-sparse attention RTL (Mistral sliding-window) + MLIR sparse-pattern lowering + llama.cpp Mistral-7B. Saturn FU now stays (carries over from Y1 + Y2); the orthogonal piece is the algorithmic shift to block-sparse attention.
- Stronger compiler novelty.
- Faculty: Han, Krishna, Abdelfattah.

**S2 — Spec decode on Saturn** [Y3 ALTERNATIVE, preserved]
HW + compiler for speculative decoding. Saturn spec-verify FU + Exo scheduling. SPAD precedent is simulator-only — open RTL slot. Hard to land in 12 months solo; may need to be Y3 or later.

## Do NOT pursue as primary (preserved 2026-05-16 PM)

- vfexp as standalone novelty: closed gap upstream (`ggml_v_expf_m2`).
- Q1 NVFP4-on-Gemmini: Berkeley overlap with Hansung Kim's `gemmini-mx` (systolic; different from Saturn vector — but still email Hansung Kim before any Gemmini work).
- Q3 Dynamic activation quant: 5+ competing recent HW/algo papers.
- S3 Prefill/decode disagg: infeasible at solo 12-mo pace; SPAD is simulator-only.
- C1 Triton-CPU-RVV: weak co-design credibility; Terapines/SpacemiT 12+ months ahead.
- C3 IREE-RVV: SiFive owns this lane in production; novelty surface near zero.

## Cross-cutting findings updated 2026-05-16 PM

Building on the original cross-cutting findings above:
7. **Upstream RVV softmax + fused FA are filled** as of mid-2026 — `ggml_v_expf_m2` and SpacemiT VLEN=1024 FA. Any "fill the gap" pitch is dead. Differentiation must be a level above: per-stage precision routing, VLEN-agnostic codegen, compiler co-design.
8. **VMXDOTP (Benini DATE 2026)** is now the dominant concurrent-work hazard along with VEXP — same group, RVV-customs-for-ML cadence is high. Watch arXiv weekly through MLArchSys submission.
9. **FuseMax IEEE Micro 45(4) 2025 retrospective** adds no new precision configs or compiler integration — methodology-only retrospective. Mixed-precision dimension stays open.
10. **Decoder-only LLM workloads with KV-cache growth** are unevaluated by FuseMax (BERT/T5/XLM/TrXL only). Llama-3.2-1B decode is a clean differentiation axis.

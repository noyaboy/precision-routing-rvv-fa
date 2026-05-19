# Y3 Speculative Decoding on Saturn — Scoping Sketch (2026-05-19)

Y3 direction locked 2026-05-19 (project memory). One-line: "spec
decode on Saturn → ISCA/ASPLOS'29, orthogonal to Y2's hetero-
precision angle." This doc fills in the formulation, the DSL
extension shape, the gate experiment, and the open questions for
2027/2028 pickup — symmetric to `y2_hetero_precision_routing.md`.

**Key landscape correction (verified 2026-05-19 via web search):**
project memory's "SPAD precedent is simulator-only" is
misattributed — SPAD (arXiv:2510.08544) is actually "Specialized
**Prefill And Decode**" (prefill/decode disaggregation), not spec
decode. The real spec-decode HW precedent is **HADES** (arXiv:
2412.19925, Dec 2024), described as the *first* HW-level spec
decode accelerator for LLMs (verification 6.99× faster than A100,
7.74× faster than A6000). Y3 framing must differentiate from
HADES, not SPAD.

## 1. Problem statement

Spec decode (Leviathan ICML'23, Chen DeepMind'23) is the dominant
inference-time LLM speedup: a small draft model proposes K
candidate tokens autoregressively, the target model verifies all
K in one parallel forward pass, the longest matching prefix is
accepted. Expected speedup = K × accept_rate, typically 2-4× on
real workloads with reasonable draft models.

HW support is fragmented and closed-source. HADES is the only
public LLM-spec-decode HW precedent (arXiv:2412.19925, late 2024)
— a standalone accelerator. SPEQ (self-spec via bit-shared
quantized draft) is more algorithm-focused. SpecMamba (FPGA,
Mamba-only) is a different model family.

**No public spec-decode HW work exists on RVV or any open vector
ISA.** The opportunity space: integrate spec-verify into the
Saturn vector unit (not a standalone accelerator), expose it via
Exo as a compiler-scheduled batched-verify primitive, and stack
it on top of Y1's precision-routing FU + Y2's per-layer policy —
so Y3 inherits the precision-routed compute pipeline rather than
duplicating it.

Y3 question: **does Saturn-integrated spec-verify (vector-unit FU
+ Exo schedule + KV rollback machinery) deliver ≥1.5× end-to-end
decode speedup vs the Y2 per-layer-policy baseline on Llama-3.2-
1B/3B, at iso-accuracy?**

## 2. Y1 + Y2 → Y3 delta (composition matters)

| Layer | Y1 (MLArchSys'27) | Y2 (MICRO'28) | Y3 (ISCA/ASPLOS'29) |
|-------|-------------------|---------------|---------------------|
| Saturn RTL | 4 precision-routing FU lanes | reused | **+ spec-verify FU + KV rollback HW** |
| Exo pass | static precision routing | + per-layer policy | **+ batched-verify schedule** |
| Llama integration | single-recipe inference | per-layer-policy inference | **+ draft-model spec, K-batched verify** |
| Eval | gem5 single-kernel | gem5 1B decode | **gem5 1B + FireSim 1B/3B end-to-end** |

Y3's *novelty* is the spec-verify FU + KV rollback machinery +
the Exo batched-verify lowering. Y1 + Y2 substrate is reused as-
is; the per-layer policy from Y2 applies to *each verify position*
in the K-batched forward pass without modification.

**The composition framing is critical** — without Y1+Y2, Y3 is
"yet another spec decode HW paper after HADES." With Y1+Y2, Y3
is "a 3-paper architecture demonstrating compiler-scheduled
precision-routed spec-decode on open RVV silicon" — a coherent
multi-paper trajectory landing one PhD application cycle.

## 3. Formulation

### 3.1 Spec decode recap (linear, K candidates)

```
1. Draft model M_d generates K candidate tokens autoregressively
   from current context: t_1, t_2, ..., t_K.
2. Target model M_t runs ONE parallel forward pass on positions
   [t_0, t_1, ..., t_K], producing logits at each position.
3. Acceptance: for i = 1..K, if M_d's sampled t_i is consistent
   with M_t's distribution at position i-1, accept; else stop.
4. Accepted prefix L (longest matching) updates the KV cache;
   rejected tokens' provisional KVs are rolled back. Sample one
   bonus token from M_t at position L from rejected distribution.
```

Expected tokens per target call: 1 + L_avg = 1 + (K × accept_rate).
For Llama-3.2 with TinyLlama-1.1B draft + Llama-3.2-3B target and
K=5, literature reports L_avg ~ 3.5 → ~3.5× decode tokens per
target step.

### 3.2 Tree-spec generalization (optional Y3 scope)

Medusa / EAGLE / SpecInfer extend linear-K to a *tree* of
candidate sequences. M_d emits a candidate tree (e.g. top-2 at
each of K positions = 2^K paths); M_t verifies all leaves in one
parallel forward pass. Acceptance walks down the tree picking the
deepest matching branch. Expected L_avg is higher (4-6 typical)
but per-call compute is also higher (~2× for top-2 K=5 tree).

Y3 scope decision: **linear-K only for the gate**, tree-spec as
optional camera-ready ablation. Tree-spec is more contested
(Medusa/EAGLE/SpecInfer all competing) and the HW story is the
same modulo verify-pass batching geometry.

### 3.3 What's actually HW-novel

| Component | SW-only is enough? | Why Saturn HW helps |
|-----------|-------------------|---------------------|
| Draft model M_d forward pass | Yes — runs as a normal small-model decode using Y1+Y2 stack | No new HW; reuses Y1+Y2 |
| Target model M_t K-parallel verify | **Saturn vector-unit advantage** — K positions × precision-routed kernels in one batched pass | New batched-verify @instr scheduling; vector lanes parallelise K cleanly |
| Acceptance mask emission | SW comparison loop | Custom `vspec.verify` FU: K-position parallel argmax-compare + mask emission |
| KV rollback | SW invalidation loop (expensive per reject) | **KV-rollback HW** — provisional-KV bit per slot, atomic invalidate-on-reject |
| Bonus-token sampling | SW from rejected distribution | reuse existing FP32 softmax accumulator (Y1) |

Two new Saturn customs proposed:
- `vspec.verify.v vd, vs1, vs2` — K-position parallel sample-vs-target compare → bit-mask in vd.
- `vspec.kvflush.v vd, vs1` — atomic invalidate KV slots in vd flagged by mask in vs1.

Both fit Saturn's `FunctionalUnitFactory` plug-in pattern (same
as Y1's 4 customs). Estimated incremental area: ~0.05 mm² at 16nm
(both are small — compare + mask + small CAM for KV slot index).

### 3.4 Composition with Y2 per-layer policy

Each verify position runs the same per-layer precision policy
π(l) that Y2 locks. The verify pass is L specialised kernels
(one per transformer layer) × K positions each. From the Exo
schedule's perspective, verify-batch is *one extra outer loop*
around the per-layer-policy kernel. No precision-policy changes
needed for Y3.

## 4. DSL / Exo angle

### 4.1 What Y3 adds to the Exo pass

Y1 pass: `lower_layer(precision) → ExoProc` emits one kernel.
Y2 pass: same, called per-layer with per-layer precision arg.
Y3 pass: **`lower_verify(precision, K, tree_shape)`** — emits
a K-batched verify kernel using the same per-layer policy.

Sketch:

```python
def lower_verify(precision: PerLayerConfig, K: int, tree: TreeShape) -> ExoProc:
    p = fa_kernel_verify_naive(K)         # K positions parallel
    p = apply_per_layer_policy(p, precision)
    # Wire spec-verify FU at acceptance stage:
    p = replace(p, "_ = sample_compare(...)", saturn_vspec_verify)
    p = replace(p, "_ = kv_invalidate(...)", saturn_vspec_kvflush)
    return simplify(p)

# Builds N specialised verify kernels (one per layer × K config).
verify_kernels = [
    lower_verify(policy[l], K=5, tree=Linear())
    for l in range(L)
]
```

The Y3 contribution shape mirrors Y2's: scheduling-DSL machinery,
not a new spec-decode algorithm. The "spec-decode algorithm" is
borrowed from Leviathan/Chen (linear-K) or Medusa/EAGLE (tree);
Y3 supplies the HW + compiler lowering onto Saturn.

### 4.2 What does NOT change

- Y1's 4 precision-routing FU lanes
- Y2's per-layer policy attribute
- The §6 (Y1) fused-FA kernel shape
- FireSim eval harness (Y2 Phase A-D already covers per-layer;
  Y3 reuses the same FireSim path with the additional `vspec`
  customs)

### 4.3 Why "compiler-scheduled" framing

HADES (and any closed-HW spec-decode accelerator) is a
black-box hardware contribution. Y3's compiler-scheduled framing
makes the contribution "the Exo extension that takes a
spec-decode algorithm + per-layer precision policy + draft model
and emits N specialised batched-verify kernels onto the shared
Saturn lanes." This positions Y3 as compiler co-design, not HW
black-box — same framing principle as Y2.

## 5. Search / scheduling machinery

### 5.1 Draft model choice

Three options:
1. **Target-family small (e.g. TinyLlama-1.1B as draft for
   Llama-3.2-3B target)** — standard practice; well-understood
   accept rates; requires running two models in memory.
2. **Self-spec via Y2 layer-drop policy** — draft is the same
   model with some layers skipped (cf. SPEQ pattern). Saves
   memory; potentially lower accept rate; tight integration with
   Y2 policy.
3. **Medusa/EAGLE-style learned-head** — fine-tuned multi-head
   added to target. Highest accept rate; requires training step
   not in our scope.

Recommend **option 1 for the gate** (cleanest comparison to HADES
+ literature baselines); **option 2 as camera-ready ablation**
(showcases Y2 stack reuse).

### 5.2 K (verify batch size) sweep

Compile-time grid: K ∈ {3, 5, 7, 10}. Measure
accept_rate(K) × cycles_per_verify(K) trade-off on Llama-3.2-1B
(gate) and -3B (camera-ready). Pick the K-knee.

Search cost: 4 K values × ~10 min gem5 per (K, decode-trace)
config ≈ overnight sweep on the existing i9-13900 host.

### 5.3 Tree shape (deferred to camera-ready)

Linear-K only for the gate. If gate passes and camera-ready
needs a stronger result, add Medusa-style top-2 tree with
budget K=5 (32 paths). Saturn lane batching handles tree
verify natively (one vse for all leaves), but the Exo schedule
needs a tree-aware lowering pass — non-trivial ~2 weeks work.

## 6. Gate experiment (Q3/Q4 2027, post-MICRO'28 submission)

Single decision point: **does Saturn-integrated spec-verify beat
the Y2 per-layer-policy baseline by ≥1.5× end-to-end decode tok/s
on Llama-3.2-1B with TinyLlama-1.1B draft, at iso-accuracy?**

### 6.1 Setup

- Target: Llama-3.2-1B (16 layers, GQA, head_dim=64; verified
  2026-05-19 against HF config.json). Camera-ready ablation:
  Llama-3.2-3B (28 layers, head_dim=128).
- Draft: TinyLlama-1.1B (22 layers, head_dim=64) — same tokenizer
  family as Llama-3.2; commonly paired in spec-decode literature.
- Eval: WikiText-2 ppl + C4 ppl (sanity — spec decode is
  *lossless* by construction; perplexity should not change vs Y2
  baseline beyond floating-point noise); accept_rate + L_avg
  measured directly on 500 prompts × 256 generated tokens.
- HW: Y1+Y2 substrate + spec-verify FU stub + KV rollback HW
  stub. Same gem5 RiscvO3CPU + DDR3-1600 as Y1/Y2 for gate; add
  FireSim cross-validation post-gate.
- Iso-accuracy threshold: spec decode is provably lossless given
  exact target sampling, so the gate is **pure speedup** at the
  same Δppl as Y2 baseline. No precision-accuracy trade-off.

### 6.2 Decision tree

| Result | Action |
|--------|--------|
| ≥1.5× tok/s, lossless | **Commit Y3.** Begin full ISCA/ASPLOS'29 paper draft. Add tree-spec + 3B ablation for camera-ready. |
| 1.2-1.5× tok/s, lossless | **Tighten.** Try (a) bigger K, (b) Medusa-tree, (c) self-spec via Y2 policy, (d) Y2 stack-integration deeper (e.g. spec-pass uses lower-precision draft via Y2 layer-drop). |
| <1.2× tok/s | **Fall back to FireSim-validation-only paper** (validation of HADES-style numbers on open RVV silicon — narrower but publishable). |

### 6.3 What this gate avoids

Same logic as Y2 §6.3: gate exists to *kill Y3 cheaply* before
the 12-month investment if the Saturn-integration doesn't carry
the 1.5× claim. Without it, Y3 risks negative result at
ISCA/ASPLOS'29 review with no fallback before PhD app deadlines.

### 6.4 Timing rationale

Y3 gate is Q3/Q4 2027 (Sep-Dec 2027) because:
- MICRO'28 submission lands Apr 2028 — Y2 paper-writing absorbs
  spring 2028.
- Y3 gate must complete before Y2 paper drafting starts in
  earnest so Y3 work can begin Jan 2028 if gate passes.
- ISCA/ASPLOS'29 deadlines: ASPLOS'29 likely Sep-Oct 2028
  abstract / Mar 2029 paper; ISCA'29 likely Aug 2028 abstract /
  Nov 2028 paper. Both feasible from Y3 gate-pass Jan 2028.
- **Primary venue: ISCA'29** (wider HW audience for spec-decode);
  **ASPLOS'29 backup** if ISCA'29 rejects (resubmit pattern
  common at this level).

## 7. Differentiation from concurrent work

### 7.1 HADES (arXiv:2412.19925) — direct precedent

Differentiate on:
- **Open RVV vs closed HW** — HADES is a standalone accelerator
  on (likely proprietary) silicon; Y3 is on Berkeley Saturn,
  open Chisel, public artifact.
- **Vector-unit-integrated vs standalone** — Y3 plugs into
  Saturn's existing vector lanes via 2 small customs; HADES
  appears to be a separate datapath. Integration matters for
  area + power on actual SoC.
- **Compiler-scheduled vs algorithm+HW co-design** — Y3
  contribution is the Exo batched-verify scheduling DSL; HADES
  is HW + algorithm without an explicit compiler-DSL story.
- **Multi-paper stack composition** — Y3 sits on Y1's precision
  routing + Y2's per-layer policy. HADES is a single-paper
  contribution.

One-paragraph diff vs HADES in §1 of Y3 paper is non-negotiable.

### 7.2 SPEQ (self-spec quantized) — moderate overlap

SPEQ extracts a quantized draft from the target via bit-shared
quantization. Y3's "self-spec via Y2 layer-drop" (option 2 in §5.1)
is structurally similar but uses Y2's compiler-scheduled per-layer
precision instead of bit-shared quant. Differentiation:
- SPEQ contribution is the quantization algorithm + HW pair; Y3
  contribution is the spec-verify FU + KV rollback + compiler
  scheduling. Different layer of the stack.
- Y3's self-spec ablation is a *parallel option*, not the
  headline. SPEQ is a competing data point, not a scoop hazard.

### 7.3 SpecMamba (arXiv:2509.19873) — different model family

FPGA accelerator for Mamba (state-space model) with spec decode.
Different model family from transformer-spec-decode; minimal
direct overlap. Cite as related FPGA-spec-decode work.

### 7.4 Medusa / EAGLE / SpecInfer (tree-spec algorithms)

Algorithm contributions; we integrate as optional camera-ready
ablation. No competition on the algorithm side; Y3 supplies the
HW + compiler substrate.

### 7.5 Han lab (sparse/streaming attention) — moderate adjacency

Han lab's streaming-attention work doesn't directly target spec
decode but the per-token-policy framing is adjacent. If they
publish a spec-decode variant during Y3 window, differentiate on
the Saturn-vector-integration + compiler-scheduled axis. Same
mitigation as Y2: email before paper-draft circulation.

### 7.6 Benini group (RVV custom instructions)

VMXDOTP (DATE'26) is the most recent. Benini group consistently
ships RVV-customs-for-ML papers. **Plausible Benini spec-decode
sequel for HPCA'28/MICRO'29.** If they publish before Y3
submission, differentiate on the Y1+Y2 stack composition (they
don't have the precision-routing substrate) and the open-Saturn
substrate (Spatz is also open but smaller community).

## 8. Hazards

| Hazard | Probability | Impact | Mitigation |
|--------|-------------|--------|------------|
| HADES has already established "first HW spec decode" framing — Y3 second-mover risk | High | Position pivot needed: from "first HW spec decode" to "first open-RVV + compiler-scheduled + stack-composed spec decode" | Differentiation must be in abstract from day one. Spend extra writing budget on §1 diff vs HADES. |
| Y3 1.5× speedup gate fails on Llama-3.2-1B (too small for spec-decode to shine; accept rates may be low for small models) | Medium | Fall back to 3B or 7B gate; FireSim-validation-only paper if all sizes fail | Run pilot accept-rate measurement on TinyLlama→1B and TinyLlama→3B before locking gate model. |
| Benini group or Han lab publishes RVV spec-decode in 2027/2028 | Medium | Scoop on a key axis | Email both before paper circulation; lean harder on Y1+Y2 stack composition (unique to us) |
| KV rollback HW is more complex than estimated (~0.05 mm²) | Low-medium | Area budget pressure; may need to drop or simplify | Prototype the FU in Chisel early (early 2028); validate area via Yosys synthesis as in Y1 track I |
| Accept rate workload-variance: WikiText-2 + C4 don't represent realistic chatbot traces | Medium | Reviewer pushback on eval | Add a ShareGPT / OASST trace eval for camera-ready (not gate) |
| 1B Llama too small to show meaningful spec-decode gain | Medium | Gate measurement may underestimate Y3 value | Run 1B gate AND 3B pilot in parallel; promote 3B to gate model if 1B is borderline |
| Tree-spec deferred from gate may become required by reviewer norms by 2028 | Medium | Camera-ready scope creep | Budget 3 weeks for tree-spec implementation in camera-ready phase; don't delay submission |
| Lossless guarantee broken by FP non-determinism (target argmax may not be stable across float-rounding paths) | Low | Spec-decode is no longer provably lossless; perplexity claim weakens | Test with `argmax_strict` mode (no tie-breaking); document any divergence as a known caveat |

## 9. Open questions for 2027/2028 pickup

When Jan 2028 Y3 pickup starts (post-MICRO'28 submission), these
need answers before any code lands:

1. **Draft model: target-family small (TinyLlama) vs self-spec
   (Y2 layer-drop) vs Medusa-head learned?** §5.1 recommends
   target-family for gate. Confirm before downloading TinyLlama
   or training Medusa heads.

2. **K knee: 3, 5, 7, 10?** §5.2 sketches the sweep; first task
   in pickup is the overnight 4-K-value run on the existing host.

3. **Tree vs linear K for the gate?** §3.2 + §5.3 recommend linear
   for gate, tree for camera-ready. Confirm based on accept rate
   results — if linear-K accept rate is <60 % at K=5, tree becomes
   necessary for the 1.5× target.

4. **Y2 stack reuse depth?** Y3 inherits the per-layer policy as-
   is (§3.4); but Y3 might benefit from running the draft model
   at lower precision than the target (e.g. draft = uniform-NVFP4,
   target = per-layer-policy). Decide before locking the spec-
   verify @proc shape.

5. **KV rollback HW vs SW-only?** Section 3.3 proposes HW
   rollback via a provisional-KV bit; SW rollback is simpler but
   adds ~50 cycles per reject. If accept rate is high (>80 %),
   SW may be acceptable. Measure first.

6. **Lossless or sampling-mode?** Spec decode is provably lossless
   for argmax sampling; lossy variants (typical, top-k, top-p) are
   common but break the lossless guarantee. Decide if Y3 supports
   only argmax (cleaner story) or also sampling (broader use).

7. **Email targets and timing**: HADES authors (technical-courtesy
   + acknowledge prior work), Benini group (same as Y2 — pre-empt
   scoop), Han lab (per-Y2 guidance). Send schedule TBD by Y3 gate
   outcome.

8. **3B or 7B for camera-ready?** Y3 gate is on 1B; camera-ready
   ablation could be 3B (smaller, aligned with Y2) or 7B (larger,
   reviewer-pleasing). Decide based on Q3 2027 reviewer norm
   trajectory.

## 10. Cross-references

- Y1 → Y2 → Y3 composition: this doc §2 table.
- Y1 paper §5.5 (Exo precision-routing pass that Y3 extends):
  `paper_draft.md:589-654`.
- Y2 scoping (per-layer policy that Y3 reuses):
  `paper/y2_hetero_precision_routing.md`.
- FireSim validation infrastructure (Y2 Phase A enables Y3 too):
  `paper/y2_firesim_prep.md`.
- Saturn FU plug-in pattern (Y3's `vspec.verify.v` +
  `vspec.kvflush.v` follow the same `FunctionalUnitFactory`
  pattern as Y1's 4 customs): see Y1 §5.3 in `paper_draft.md` +
  `saturn-fu/` standalone project structure.
- Spec-decode HW precedents (verified 2026-05-19):
  - HADES — arXiv:2412.19925, Dec 2024 (direct precedent)
  - SPEQ — bit-shared self-spec
  - SpecMamba — arXiv:2509.19873, FPGA Mamba-only
  - From Quarter to All — arXiv:2510.18525, FP exponent remapping
  - Traversal Verification — arXiv:2505.12398, tree-decode
  - Saguaro — algorithm-only, +30 % over baseline spec
- Spec-decode algorithm precedents:
  - Leviathan ICML'23 (original linear-K)
  - Chen DeepMind'23 (parallel formulation)
  - Medusa, EAGLE, SpecInfer (tree variants)

## 11. What this doc deliberately does NOT decide

- The exact draft model. §5.1 leans target-family small; verify
  in Jan 2028 pickup.
- K knee. §5.2 sketches a 4-value sweep; first pickup task.
- Tree-spec details. Deferred to camera-ready.
- Specific spec-verify FU pipeline depth. RTL sketch in 2028.
- Y3 paper title / abstract. Premature.
- ISCA'29 vs ASPLOS'29 specific submission. §6.4 names ISCA'29
  primary, ASPLOS'29 backup; reconfirm based on deadline calendars
  in Q3 2028.
- Hardware purchase. Still zero-cost per `feedback-prefer-zero-
  cost`. Saturn + FireSim (Y2 Phase B-D advisor-borrow) covers
  Y3.

This doc is a planning artifact, not a paper draft. When Y3 gate
passes (or fails), it gets archived; the Y3 paper draft lives
elsewhere.

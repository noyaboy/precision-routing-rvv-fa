# Y2 Session 1 — early-pickup notes (2026-05-19)

Side-project dormancy default per `y1_warmup.md:49-51` was opted-out this
session; Y2 prep started ~3 months ahead of the Aug 2026 window the doc
schedules. Single deliverable: download + integrity-verify the
Llama-3.2-1B-Instruct F16 GGUF that the Q2 band sweep (per
`y2_hetero_precision_routing.md` §6.1 + §9 Q2) needs as its substrate.

## What landed

- **File**: `models/Llama-3.2-1B-Instruct-f16.gguf` (2,479,595,360 B / 2.48 GB).
- **Source**: `bartowski/Llama-3.2-1B-Instruct-GGUF` on HuggingFace
  (community redistribution; not Meta-official — gated repo would need
  HF token + `convert_hf_to_gguf.py`).
- **SHA256**: `1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146`.
- **HTTP Content-Length** matched final file size exactly. No partial transfer.

## Metadata smoke-test (gguf-py)

All Y2 doc §6.1 + `precision_config.md:62` claims verified against the
GGUF header (147 tensors, llama-bpe tokenizer, llama-3.2 license):

| Field | Expected | GGUF |
|---|---|---|
| `general.architecture` | `llama` | `llama` ✅ |
| `llama.block_count` | 16 layers | 16 ✅ |
| `llama.attention.head_count` | 32 | 32 ✅ |
| `llama.attention.head_count_kv` | 8 (GQA) | 8 ✅ |
| `llama.attention.key_length` / `value_length` | head_dim=64 | 64 ✅ |
| `llama.embedding_length` | hidden=2048 | 2048 ✅ |
| `llama.context_length` | (128K window) | 131072 |
| `general.file_type` | F16 (MOSTLY_F16 = 1) | 1 ✅ |

## What was deliberately NOT done

- **Full RISC-V decode smoke test**: would require either (a) gem5 SE mode
  (~1-2 hr load + decode for 1B F16 model under simulation; canonical
  but slow), or (b) a new x86 llama-cli build (one-time ~15 min compile,
  off-Saturn). Neither is on the Q2 critical path; deferred to the
  band-sweep harness build session.
- **Quantization to NVFP4 K/V**: depends on the Saturn-FU `vfconv` lanes'
  llama.cpp RVV integration, which itself is part of the Y2 deliverable
  set below.

## Concrete next-session prereqs (for Q2 band sweep)

Ordered by dependency. Each item is a self-contained session-scoped
deliverable; combined effort ~3-4 sessions before the Q2 sweep can
actually run.

1. **WikiText-2 perplexity harness inside our `llama.cpp` RVV build.**
   - Add `perplexity` binary build target if not already in
     `build/llama-rv64gcv/`. (llama.cpp ships `tools/perplexity/perplexity.cpp`
     upstream — likely already builds; need to verify against our local
     patches per project-memory's `__GNUC__ >= 14` gates.)
   - Test wire: WikiText-2 raw text → llama-cli's tokenizer → log-perplexity
     over a 256-sequence calibration set + 1000-sequence eval set per Y2
     §6.1 protocol.
   - Validation: uniform-F16 perplexity on Llama-3.2-1B should land
     ~9.7-10.0 on WikiText-2 (matches ARCQuant baseline numbers; if
     materially off, GGUF or tokenizer is wrong).

2. **C4 perplexity harness** (same machinery as #1, different dataset
   loader). Defer if time-constrained — WikiText-2 alone is enough for
   the band-size sub-sweep.

3. **NVFP4 K/V quantization path through llama.cpp RVV decoder.**
   - Uniform-NVFP4 variant first (the Y1 paper's already-locked recipe).
   - Hooks our existing `saturn-fu/` `VFConvNvfp4Bf16Lane` (bit-exact
     RTL, 0/4096 ChiselTest mismatches) via the `vfconv.nvfp4.bf16.v`
     intrinsic emitted by our Exo pass.
   - Calibration: per-channel KV scales computed offline on WikiText-2
     [0:256] (Y2 §6.1 protocol).
   - Sanity gate: uniform-NVFP4-K/V perplexity Δ vs uniform-F16 should
     be ≤0.5 ppl per ARCQuant baseline. If materially worse, calibration
     code has a bug.

4. **Per-layer policy injection mechanism.** This is the actual Y2
   contribution surface — the Exo schedule attribute that Y2 §4.1
   sketches. Two equivalent forms (pick later):
   - **N specialized kernels**: emit a separate llama.cpp RVV kernel
     per layer with that layer's K/V dtype baked in. Static dispatch
     via layer-id parameter at the C-API boundary.
   - **One parameterized kernel with switch**: single kernel reads a
     per-layer-id table of dtype flags at call entry. Cheaper to
     iterate during sweep; slightly higher runtime overhead.

5. **Q2 band sweep execution.** With items 1+3+4 above, the sweep
   itself (6 configs × ~10 min gem5 fwd-pass ≈ 1 hr) is a single
   overnight gem5 batch on the existing i9-13900 host.

## Things to NOT do yet

Per Y2 §11 "what this doc deliberately does NOT decide":

- The actual per-layer policy. That's the sweep *output*, not a
  pre-commitment.
- The Y2 paper title / abstract. Premature; wait until gate passes.
- Llama-3.2-3B GGUF download. Camera-ready material, not gate material.
- FireSim Phase A. The Y2 doc §9 Q7 puts Phase A in the Nov 2026 -
  Mar 2027 Y1 polish window, not now.
- Email Ragan-Kelley + Hansung Kim. Y2 doc §9 Q6 says "send NOW" for
  these two; that's an external/user action, not assistant-executable.

## Cross-references

- Y2 scoping doc: `paper/y2_hetero_precision_routing.md` (~17 KB).
- Y2 §6.1 setup: gem5 RiscvO3CPU + DDR3-1600 + WikiText-2 + C4.
- Y2 §9 Q2: band sweep design (6 configs).
- Y1 locked recipe: `paper/precision_config.md`.
- Y1 RVV decoder integration patterns: `microbench-fa/` build scripts;
  `paper/track_j4_results.md` for the GCC 13.2 vsetvli workaround.
- Project memory: `feedback-rvv-asm-vsetvli` (re-check vsetvli stream
  before believing perplexity numbers — same hazard surface as Y1 J-4.5b).

## Baseline perplexity (locked 2026-05-19)

Uniform-F16 baseline for the Q2 band sweep gate is now measured:

| Metric | Value |
|---|---|
| Model | `Llama-3.2-1B-Instruct-f16.gguf` (SHA above) |
| Eval | WikiText-2 `wiki.test.raw`, 141 chunks |
| Context | 2048 (matches Y2 doc §6.1 protocol) |
| Threads | 32 (x86 AVX2 + OpenMP) |
| **PPL** | **11.9048 ± 0.0843** |
| Wall time | ~60 min on 32-core x86 (28.12 s/pass × 141) |

**Caveat on absolute number**: this is the **Instruct** variant, not the
base model. Llama-3.2-1B base ppl on WikiText-2 typically lands 9-10
(matches ARCQuant's reported baseline); Instruct adds ~2-3 ppl due to
instruction-tuning skew. The Y2 gate is a **Δppl signal** across
per-layer configs vs this uniform-F16 baseline — the choice of base vs
Instruct doesn't affect the Δ ranking. If reviewer-comparability to
ARCQuant becomes load-bearing later, swap to `meta-llama/Llama-3.2-1B`
(base, gated, needs HF token + `convert_hf_to_gguf.py`).

## x86 toolchain provisioned this session

- **Build dir**: `build/llama-x86/` (cmake config + `llama-perplexity`
  target built). Separate from `build/llama-rv64gcv/` — RISC-V cross-
  compile setup untouched.
- **Config**: `-DCMAKE_BUILD_TYPE=Release -DGGML_RVV=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON`.
- **System detected**: x86_64, SSE3 / SSSE3 / AVX / AVX2 / F16C / FMA / BMI2
  / OpenMP. AVX-512 not used (likely Zen2 host — verify if numerics
  matter later).
- **Why an x86 build is fine for perplexity**: perplexity is a
  deterministic function of weights × tokens, independent of
  microarchitecture. Y2's RISC-V/gem5 path is for *cycles*, not ppl.

## Datasets provisioned

- `datasets/wikitext-2-raw/wiki.{test,valid,train}.raw` (Smerity
  mirror, 4.7 MB zip, unzipped to 13 MB). Test split = 1.29 MB, 241K
  words, ~289K Llama-3 tokens.

## Session-state for next pickup

- `models/`: now contains `qwen2.5-0.5b-instruct-{q4_k_m,q8_0}.gguf` +
  `Llama-3.2-1B-Instruct-f16.gguf`.
- `build/`: now contains `llama-rv64gcv/` (unchanged) + `llama-x86/` (new).
- `datasets/`: new dir, holds `wikitext-2-raw/`.
- `paper/y2_session1_notes.md` (this file) is the only new paper-tree
  artifact.
- **Baseline ppl 11.9048 ± 0.0843** is the gate reference; downstream
  per-layer policy configs need only beat this within `ε = +0.1 ppl`
  to be considered iso-accuracy (Y2 §6.1).

## Dormancy reminder

`y1_warmup.md` lists Aug 2026 - Oct 2026 as the Y2 scoping pickup window.
This session opted out 3 months early. If main-group bandwidth tightens,
the natural fallback is to pause Y2 prep and resume on the doc's
schedule — items 1-5 above are independent enough to pick up anytime
without rebase pain.

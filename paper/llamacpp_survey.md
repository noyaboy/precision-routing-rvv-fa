# llama.cpp RVV path survey (2026-05-16, **rewritten PM after upstream-state recheck**)

Earlier sections of this file claimed `ggml_vec_soft_max_f32` had "no RVV path." **That claim is dead.** Mid-session 2026-05-16 PM, when going to instrument the softmax kernel for the Mo 2 measurement, the kernel was opened and an RVV path was found right there. Multiple other gaps the original survey flagged are also closed. Updated state below.

## Closed gaps (NEW finding 2026-05-16 PM)

1. **`ggml_vec_soft_max_f32` HAS an RVV path** at `ggml/src/ggml-cpu/vec.cpp:600-608`, behind `#elif defined(__riscv_v_intrinsic)`. Implementation calls `ggml_v_expf_m2()` defined in `vec.h:1342-1378` — a polynomial RVV expf via range-reduce + degree-5 minimax + reconstruct (~12 vector ops per iteration). The function is also used by `ggml_v_silu_m2()` at `vec.h:1381-1386` and is referenced by the generic flash-attention forward path's inner loop.
2. **Generic fused flash-attention** (`ggml_compute_forward_flash_attn_ext` at `ops.h:89`, `ops.cpp:~8650`) uses online softmax with scalar `expf` for cross-tile rescaling but vectorized `ggml_vec_soft_max_f32` for the inner kernel. Generic dispatch table is at `ggml-cpu.c:1987` (`GGML_OP_FLASH_ATTN_EXT`).
3. **SpacemiT VLEN=1024 fused FA** at `spacemit/rvv_kernels.cpp:1121, 1305` — `forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16` and `forward_flash_attn_ext_f16_tiled_vlen1024_vf16`. **Hardcoded `__riscv_vlenb() * 8 == 1024` check** (`flash_attn_ext_supported_d_vlen1024_vf16` at line 37). K1-only; doesn't trigger for Saturn (VLEN=256). Their inner softmax is an adapted version of `ggml_v_expf_m2` (comment at `rvv_kernels.cpp:57`).

**Implication for the project:** The original Proposal X premise ("fill the scalar RVV softmax gap with vfexp") is dead. The Y1 direction has been pivoted to merged X+Y (mixed-precision fused FA with per-stage precision routing) — see [[project-side-project-direction]] for the new framing.

## Current RVV-touched files in llama.cpp master (May 2026)

**Generic (architecture-dispatching):**
- `ggml/src/ggml-cpu/vec.h`, `vec.cpp` — vector primitives. **vec.h:1342 ggml_v_expf_m2 (RVV polynomial expf)**. vec.h:1381 ggml_v_silu_m2.
- `ggml/src/ggml-cpu/ops.cpp` — graph ops, calls softmax kernel from generic FA online-softmax rescale loop.
- `ggml/src/ggml-cpu/ops.h` — `ggml_compute_forward_flash_attn_ext` prototype.
- `ggml/src/ggml-cpu/simd-mappings.h` — arch SIMD type/macro mappings.
- `ggml/src/ggml-cpu/simd-gemm.h` — generic GEMM via `GGML_F32_VEC_*`.
- `ggml/src/ggml-cpu/ggml-cpu.{c,cpp}` — op dispatch table (GGML_OP_SOFT_MAX line 1879, GGML_OP_FLASH_ATTN_EXT line 1987).

**RISC-V specific (arch/riscv/):**
- `arch/riscv/quants.c` (4553 lines) — quant kernels. **Note**: many functions use GCC-14+ intrinsics (`__riscv_vcreate_v_*`, segment-tuple types) that don't compile on GCC ≤ 13. Our local build has 6 functions gated with `__GNUC__ >= 14` to compile cleanly on Bootlin GCC 13.2 — patches are LOCAL, don't upstream.
- `arch/riscv/repack.cpp` (1703 lines) — quant data repacking. Has one GCC-14-only site (`vint64m2x4_t` + `__riscv_vsseg4e64`) — patched locally.
- `arch/riscv/cpu-feats.cpp` — RVV feature detection via `<asm/hwprobe.h>`.

**SpacemiT-specific (`spacemit/` subdir):**
- `spacemit/ime.{cpp,h}` — SpacemiT IME matrix accelerator interface. Confirms `GGML_OP_FLASH_ATTN_EXT` at line 941 + 1035 — routes to spacemit path if `flash_attn_ext_supported_shape_vlen1024_vf16` returns true (i.e., K1).
- `spacemit/ime1_kernels.cpp`, `ime2_kernels.cpp` — IME matrix kernels.
- `spacemit/rvv_kernels.{cpp,h}` (3178 LOC) — RVV softmax + fused FA kernels for K1. **Note for our work**: their `rvv_kernels.cpp:57` comment says "Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax" — they reuse the polynomial expf from vec.h.
- `spacemit/repack.{cpp,h}` — SpacemiT data repack.
- `spacemit/spine_mem_pool.{cpp,h}` — memory pool for IME.
- `spacemit/ime_env.{cpp,h}` — env/config.

## Q4_K_M dispatch reality (relevant for any K-quant-based Mo 2)

`ggml_vec_dot_q4_K_q8_K` at `arch/riscv/quants.c:1321` has **only** `__riscv_xtheadvector` (XuanTie's pre-1.0 RVV) fast path. On standard `rv64gcv` builds (Saturn, generic RVV 1.0), it falls through to the **generic scalar** path. So upstream's Q4_K matmul on Saturn is scalar — not vectorized. Our Mo 2 microbench should either (a) avoid Q4_K models and use Q8_0 / Q4_0 instead (which have working `__riscv_v` paths), or (b) write a Q4_K RVV 1.0 kernel as part of our contribution.

The cross-compiled `llama-cli` we built (tonight, with Q8_0 model) exercises the working RVV 1.0 path. Q4_K_M model is also downloaded but should be used carefully — runs but uses scalar matmul.

## Mo 2 measurement protocol (replaces old vfexp-fraction question)

The new Mo 2 question is **"NVFP4 K/V reduces decode memory bandwidth ≥30% vs FP16 K/V?"** Detailed protocol in `measurement_plan.txt`. Briefly: gem5 SE TimingSimpleCPU + cache hierarchy, NVFP4-dequant + GEMV microbench vs FP16 GEMV baseline, BW counters from gem5 stats. **Does NOT require profiling llama.cpp end-to-end** (analytical 3.2× BW reduction from NVFP4 block-16 vs FP16 is the anchor).

The old "softmax/exp ≥10% of decode" Mo 2 question is **deleted** — based on the now-disproved premise that scalar softmax was the bottleneck on upstream RVV.

## What was right in the original survey (preserved)

- "`spacemit/rvv_kernels.cpp` — likely RVV-vectorized kernels" — now confirmed: yes, 3178 LOC of RVV-vectorized softmax + fused FA, K1-specific (VLEN=1024).
- "SpacemiT IME accelerates softmax" question — answer: **IME does matmul; SpacemiT's RVV softmax is in `spacemit/rvv_kernels.cpp` (a separate, non-IME path)**. So K1 softmax is vectorized through RVV poly expf, not through the IME accelerator.

## Updates to other artifacts (cross-references)

- `[[project-side-project-direction]]` updated 2026-05-16 PM with merged X+Y direction and closed-gap notes.
- `[[reference-key-papers-repos]]` updated with VMXDOTP, ARCQuant, Cygnus, SGLang, FuseMax IEEE Micro 2025.
- `intro_draft.txt` v2 written around new framing.
- `related_work.md` rewritten with new comparison axes.
- `measurement_plan.txt` rewritten with new Mo 2 protocol.

## Action items still open

- When pitching the paper, the framing needs to address SpacemiT IME explicitly. "First open RVV-1.0 attention stack" is dead; the correct framing is "first open per-stage mixed-precision RVV fused FA + VLEN-parametric Exo schedule, beating both upstream generic RVV FA and SpacemiT's K1-specific VLEN=1024-only FA."
- Periodic re-grep on `vec.h` / `ops.cpp` / `spacemit/` for any further closed gaps — set a cadence of monthly checks. See `[[feedback-recheck-upstream-state]]`.

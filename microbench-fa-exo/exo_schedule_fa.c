#include "exo_schedule_fa.h"
#include "exo_schedule_fa.h"


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */

#include <stdio.h>
#include <stdlib.h>


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */


#ifndef SATURN_CUSTOM_ASM_H
#define SATURN_CUSTOM_ASM_H
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* vfexp.v vd, vs2   (VS1=0x06, BF16-in / FP32-out, LMUL doubles) */
#define SATURN_VFEXP(dst, src, vl) do {                                    \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfexp.v at SEW=32/m2 — FP32 input, FP32 output. Matches the
 * bench's inline-asm shortcut (bench_fa_mixed_rvv_native.c lines
 * 226-247): vfexp.v reads register state's current SEW, and the
 * Saturn VFExpLane RTL (Track E) takes FP32 input directly. This
 * variant eliminates the FP32 -> BF16 narrow step before vfexp in
 * softmax Pass 2, saving 1 macro call + 1 vsetvli barrier per
 * SEQ_LEN/16 chunk. */
#define SATURN_VFEXP_F32M2(dst, src, vl) do {                              \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E831457\n\t"   /* vfexp.v v8, v8 */                       \
      "vse32.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9");                                             \
} while (0)

/* vfconv.fp8.bf16.v vd, vs2   (VS1=0x07, FP8-in / BF16-out, LMUL doubles) */
#define SATURN_VFCONV_FP8_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vle8.v v8, (%1)\n\t"                                                \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      ".4byte 0x4E839057\n\t"   /* vfconv.fp8.bf16.v v0, v8 */             \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.bf16.fp8.v vd, vs2   (VS1=0x08, BF16-in / FP8-out, LMUL halves) */
#define SATURN_VFCONV_BF16_FP8(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      ".4byte 0x4E841457\n\t"   /* vfconv.bf16.fp8.v v8, v8 */             \
      "vsetvli zero, %2, e8, mf2, ta, ma\n\t"                              \
      "vnsrl.wi v0, v8, 0\n\t"                                             \
      "vse8.v v0, (%0)\n\t"                                                \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v8");                                             \
} while (0)

/* vfconv.nvfp4.bf16.v vd, vs2   (VS1=0x09, NVFP4-in / BF16-out;
 * single-source variant — application multiplies by E4M3 scale
 * separately. See paper §4.3 for the fused alternative). */
#define SATURN_VFCONV_NVFP4_BF16(dst, src, vl) do {                        \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0");                                                   \
} while (0)

/* Mo 8 step 4d-5: vfconv.nvfp4.bf16.v at LMUL=4 — processes 64 NVFP4
 * nibbles per asm-volatile boundary (vs 16 at m1). Matches bench's
 * dequant_64elt_chunk LMUL=4 layout: 1 vfconv invocation per head_dim=64
 * row instead of 4 m1 invocations. The result is 64 BF16 lanes spread
 * across v0-v3 (m4 register group); we stash to DRAM and reload per
 * 16-lane chunk for the widen + scale. */
#define SATURN_VFCONV_NVFP4_BF16_M4(dst, src, vl) do {                     \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m4, ta, ma\n\t"                              \
      "vle16.v v0, (%1)\n\t"                                               \
      ".4byte 0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */           \
      "vse16.v v0, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v0", "v1", "v2", "v3");                                 \
} while (0)

/* Mo 8 step 4d-1: Reductions via static-inline helpers using intrinsics.
 *
 * The previous variant declared these as `do { ... } while (0)` macros.
 * Static inline functions let GCC inline + schedule them across
 * surrounding code without the macro's potential scope inhibitions; the
 * critical property is that NO asm-volatile is involved, so the OoO
 * core can overlap reductions with adjacent ops.
 *
 * Used by saturn_vfredmax_to_dram_f32m2 / saturn_vfredusum_to_dram_f32m2.
 * The BF16<->FP32 widen + narrow are no longer macros: their @instr
 * templates emit the intrinsic chain inline at the call site. */

static inline float saturn_vfredmax_f32m2_helper(float init,
                                                 vfloat32m2_t src,
                                                 size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredmax_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static inline float saturn_vfredusum_f32m2_helper(float init,
                                                  vfloat32m2_t src,
                                                  size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(init, 1);
    acc = __riscv_vfredusum_vs_f32m2_f32m1(src, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

#endif /* SATURN_CUSTOM_ASM_H */

// fa_kernel_decode_naive(
//     seq_len : size,
//     qk_scale : f32[1] @DRAM,
//     Q_fp32 : f32[8, 64] @DRAM,
//     K_nvfp4 : ui16[8, seq_len, 4, 16] @DRAM,
//     K_scale : f32[8, seq_len, 4] @DRAM,
//     V_nvfp4 : ui16[8, seq_len, 4, 16] @DRAM,
//     V_scale : f32[8, seq_len, 4] @DRAM,
//     O_fp32 : f32[8, 64] @DRAM
// )
void fa_kernel_decode_naive( void *ctxt, int_fast32_t seq_len, const float* qk_scale, const float* Q_fp32, const uint16_t* K_nvfp4, const float* K_scale, const uint16_t* V_nvfp4, const float* V_scale, float* O_fp32 ) {
EXO_ASSUME(seq_len > 0);
EXO_ASSUME(seq_len % 16 == 0);
float *S_fp32 = (float*) malloc(seq_len * sizeof(*S_fp32));
float *P_fp32 = (float*) malloc(seq_len * sizeof(*P_fp32));
float *K_fp32_row = (float*) malloc(64 * sizeof(*K_fp32_row));
float *V_fp32_row = (float*) malloc(64 * sizeof(*V_fp32_row));
float *m_state = (float*) malloc(1 * sizeof(*m_state));
float *l_state = (float*) malloc(1 * sizeof(*l_state));
for (int_fast32_t h = 0; h < 8; h++) {
  for (int_fast32_t s = 0; s < seq_len; s++) {
    for (int_fast32_t kblk = 0; kblk < 4; kblk++) {
      vuint16m1_t K_nvfp4_reg;
      vuint16m1_t K_bf16_reg;
      vfloat32m2_t K_fp32_reg;
      vfloat32m2_t K_scaled;
      K_nvfp4_reg = __riscv_vle16_v_u16m1(&K_nvfp4[(h) * (seq_len * 64) + (s) * (64) + (kblk) * (16)], (16));
      SATURN_VFCONV_NVFP4_BF16(&K_bf16_reg, &K_nvfp4_reg, (16));
      asm volatile ("vsetvli zero, %0, e32, m2, ta, ma" :: "r"((size_t)((16)))); K_fp32_reg = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(K_bf16_reg, (16)), 16, (16)));
      K_scaled = __riscv_vfmul_vf_f32m2(K_fp32_reg, K_scale[(h) * (seq_len * 4) + (s) * 4 + kblk], (16));
      __riscv_vse32_v_f32m2(&K_fp32_row[16 * kblk], K_scaled, (16));
    }
    vfloat32m2_t Q_reg;
    vfloat32m2_t K_reg;
    vfloat32m2_t S_acc;
    S_acc = __riscv_vfmv_v_f_f32m2(0.0f, (16));
    for (int_fast32_t qko = 0; qko < 4; qko++) {
      Q_reg = __riscv_vle32_v_f32m2(&Q_fp32[(h) * (64) + 16 * qko], (16));
      K_reg = __riscv_vle32_v_f32m2(&K_fp32_row[16 * qko], (16));
      S_acc = __riscv_vfmacc_vv_f32m2(S_acc, Q_reg, K_reg, (16));
    }
    S_fp32[s] = 0.0f;
    S_fp32[s] = saturn_vfredusum_f32m2_helper(S_fp32[s], S_acc, (16));
    S_fp32[s] = S_fp32[s] * qk_scale[0];
  }
  m_state[0] = -1e+30f;
  l_state[0] = 0.0f;
  for (int_fast32_t so1 = 0; so1 < ((seq_len) / (16)); so1++) {
    vfloat32m2_t S_reg1;
    S_reg1 = __riscv_vle32_v_f32m2(&S_fp32[16 * so1], (16));
    m_state[0] = saturn_vfredmax_f32m2_helper(m_state[0], S_reg1, (16));
  }
  for (int_fast32_t so2 = 0; so2 < ((seq_len) / (16)); so2++) {
    vfloat32m2_t S_reg2;
    vfloat32m2_t S_shifted;
    vfloat32m2_t P_reg;
    S_reg2 = __riscv_vle32_v_f32m2(&S_fp32[16 * so2], (16));
    S_shifted = __riscv_vfsub_vf_f32m2(S_reg2, m_state[0], (16));
    SATURN_VFEXP_F32M2(&P_reg, &S_shifted, (16));
    __riscv_vse32_v_f32m2(&P_fp32[16 * so2], P_reg, (16));
    l_state[0] = saturn_vfredusum_f32m2_helper(l_state[0], P_reg, (16));
  }
  for (int_fast32_t d = 0; d < 64; d++) {
    O_fp32[h * 64 + d] = 0.0f;
  }
  for (int_fast32_t s = 0; s < seq_len; s++) {
    for (int_fast32_t vblk = 0; vblk < 4; vblk++) {
      vuint16m1_t V_nvfp4_reg;
      vuint16m1_t V_bf16_reg;
      vfloat32m2_t V_fp32_reg;
      vfloat32m2_t V_scaled;
      V_nvfp4_reg = __riscv_vle16_v_u16m1(&V_nvfp4[(h) * (seq_len * 64) + (s) * (64) + (vblk) * (16)], (16));
      SATURN_VFCONV_NVFP4_BF16(&V_bf16_reg, &V_nvfp4_reg, (16));
      asm volatile ("vsetvli zero, %0, e32, m2, ta, ma" :: "r"((size_t)((16)))); V_fp32_reg = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(V_bf16_reg, (16)), 16, (16)));
      V_scaled = __riscv_vfmul_vf_f32m2(V_fp32_reg, V_scale[(h) * (seq_len * 4) + (s) * 4 + vblk], (16));
      __riscv_vse32_v_f32m2(&V_fp32_row[16 * vblk], V_scaled, (16));
    }
    for (int_fast32_t pko = 0; pko < 4; pko++) {
      vfloat32m2_t V_reg;
      vfloat32m2_t O_reg;
      V_reg = __riscv_vle32_v_f32m2(&V_fp32_row[16 * pko], (16));
      O_reg = __riscv_vle32_v_f32m2(&O_fp32[(h) * (64) + 16 * pko], (16));
      O_reg = __riscv_vfmacc_vf_f32m2(O_reg, P_fp32[s], V_reg, (16));
      __riscv_vse32_v_f32m2(&O_fp32[(h) * (64) + 16 * pko], O_reg, (16));
    }
  }
}
free(l_state);
free(m_state);
free(V_fp32_row);
free(K_fp32_row);
free(P_fp32);
free(S_fp32);
}


/* relying on the following instruction..."
saturn_bf16_widen_f32_m2(dst,src,vl)
asm volatile ("vsetvli zero, %0, e32, m2, ta, ma" :: "r"((size_t)({vl}))); {dst_data} = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2({src_data}, {vl}), 16, {vl}));
*/

/* relying on the following instruction..."
saturn_vfmacc_vf_f32m2(dst,scalar,src,vl)
{dst_data} = __riscv_vfmacc_vf_f32m2({dst_data}, {scalar_data}, {src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfmacc_vv_f32m2(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f32m2({dst_data}, {lhs_data}, {rhs_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfmul_vf_f32m2(dst,src,scalar,vl)
{dst_data} = __riscv_vfmul_vf_f32m2({src_data}, {scalar_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfmv_zero_f32m2(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m2(0.0f, {vl});
*/

/* relying on the following instruction..."
saturn_vfredmax_to_dram_f32m2(dst,src,vl)
{dst_data} = saturn_vfredmax_f32m2_helper({dst_data}, {src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfredusum_to_dram_f32m2(dst,src,vl)
{dst_data} = saturn_vfredusum_f32m2_helper({dst_data}, {src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfsub_vf_f32m2(dst,src,scalar,vl)
{dst_data} = __riscv_vfsub_vf_f32m2({src_data}, {scalar_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vle16_m1(dst,src,vl)
{dst_data} = __riscv_vle16_v_u16m1(&{src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vle32_m2(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m2(&{src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vse32_m2(dst,src,vl)
__riscv_vse32_v_f32m2(&{dst_data}, {src_data}, {vl});
*/

/* relying on the following instruction..."
vfconv_nvfp4_bf16_v(dst,src,vl)
SATURN_VFCONV_NVFP4_BF16(&{dst_data}, &{src_data}, {vl});
*/

/* relying on the following instruction..."
vfexp_f32_v(dst,src,vl)
SATURN_VFEXP_F32M2(&{dst_data}, &{src_data}, {vl});
*/

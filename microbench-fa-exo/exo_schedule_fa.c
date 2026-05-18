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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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

/* Widen BF16 (ui16 carrier) -> FP32 via vzext.vf2 + vsll.vi 16.
 * This is the gem5-compatible widen path used in
 * bench_fa_mixed_rvv_native.c: gem5 25.1.0.1 SE mode lacks Zvfbfwma
 * and Zvfbfmin, so vfwcvtbf16.f.f.v isn't available; instead we
 * zero-extend each ui16 lane to ui32 and shift the BF16 bits into
 * the FP32 sign+exp+mantissa-upper-7 positions (BF16 == upper 16
 * bits of FP32). Input: 16 BF16 at LMUL=1; output: 16 FP32 at
 * LMUL=2. */
#define SATURN_BF16_WIDEN_F32(dst, src, vl) do {                           \
    asm volatile (                                                         \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vle16.v v8, (%1)\n\t"                                               \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vzext.vf2 v10, v8\n\t"                                              \
      "vsll.vi v10, v10, 16\n\t"                                           \
      "vse32.v v10, (%0)\n\t"                                              \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v8", "v9", "v10", "v11");                               \
} while (0)

/* Truncating narrow FP32 -> BF16 via vnsrl.wi 16. Same gem5-compat
 * reason as the widen (no Zvfbfmin). Each FP32 element's upper 16
 * bits (sign + exp + mantissa[22:16]) become the BF16 element; lower
 * 16 bits of mantissa are dropped (truncate, not round). LMUL halves:
 * 16 FP32 (M2) in -> 16 BF16 (M1) out. */
#define SATURN_F32_NARROW_BF16(dst, src, vl) do {                          \
    asm volatile (                                                         \
      "vsetvli zero, %2, e32, m2, ta, ma\n\t"                              \
      "vle32.v v4, (%1)\n\t"                                               \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                              \
      "vnsrl.wi v8, v4, 16\n\t"                                            \
      "vse16.v v8, (%0)\n\t"                                               \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                             \
      : "memory", "v4", "v5", "v8");                                       \
} while (0)

/* Reduce FP32 vector to scalar max into a DRAM-resident f32. Bounces
 * the scalar through an m1 register via vfmv.v.f init + vfredmax.vs
 * + vfmv.f.s extract. The DRAM-scalar surface keeps the @instr
 * signature uniform with the rest of the platform; step 4
 * (cycle-parity work) can replace this with a more granular variant
 * that holds the scalar in a SaturnRVV register. */
#define SATURN_VFREDMAX_F32M2(dst_scalar, src, vl) do {                    \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));             \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

/* Reduce FP32 vector to scalar sum (unordered) into a DRAM-resident
 * f32. Same bouncing pattern as the max-reduce above. Unordered
 * variant (vfredusum) over ordered (vfredsum) because softmax sums
 * already absorb FP rounding; unordered is faster and the result
 * difference is below softmax's numerical tolerance. */
#define SATURN_VFREDUSUM_F32M2(dst_scalar, src, vl) do {                   \
    vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);           \
    _acc = __riscv_vfredusum_vs_f32m2_f32m1((src), _acc, (vl));            \
    (dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);                       \
} while (0)

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
      SATURN_BF16_WIDEN_F32(&K_fp32_reg, &K_bf16_reg, (16));
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
    SATURN_VFREDUSUM_F32M2(S_fp32[s], S_acc, (16));
    S_fp32[s] = S_fp32[s] * qk_scale[0];
  }
  m_state[0] = -1e+30f;
  l_state[0] = 0.0f;
  for (int_fast32_t so1 = 0; so1 < ((seq_len) / (16)); so1++) {
    vfloat32m2_t S_reg1;
    S_reg1 = __riscv_vle32_v_f32m2(&S_fp32[16 * so1], (16));
    SATURN_VFREDMAX_F32M2(m_state[0], S_reg1, (16));
  }
  for (int_fast32_t so2 = 0; so2 < ((seq_len) / (16)); so2++) {
    vfloat32m2_t S_reg2;
    vfloat32m2_t S_shifted;
    vuint16m1_t S_bf16;
    vfloat32m2_t P_reg;
    S_reg2 = __riscv_vle32_v_f32m2(&S_fp32[16 * so2], (16));
    S_shifted = __riscv_vfsub_vf_f32m2(S_reg2, m_state[0], (16));
    SATURN_F32_NARROW_BF16(&S_bf16, &S_shifted, (16));
    SATURN_VFEXP(&P_reg, &S_bf16, (16));
    __riscv_vse32_v_f32m2(&P_fp32[16 * so2], P_reg, (16));
    SATURN_VFREDUSUM_F32M2(l_state[0], P_reg, (16));
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
      SATURN_BF16_WIDEN_F32(&V_fp32_reg, &V_bf16_reg, (16));
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
SATURN_BF16_WIDEN_F32(&{dst_data}, &{src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_f32_narrow_bf16_m2(dst,src,vl)
SATURN_F32_NARROW_BF16(&{dst_data}, &{src_data}, {vl});
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
SATURN_VFREDMAX_F32M2({dst_data}, {src_data}, {vl});
*/

/* relying on the following instruction..."
saturn_vfredusum_to_dram_f32m2(dst,src,vl)
SATURN_VFREDUSUM_F32M2({dst_data}, {src_data}, {vl});
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
vfexp_v(dst,src,vl)
SATURN_VFEXP(&{dst_data}, &{src_data}, {vl});
*/

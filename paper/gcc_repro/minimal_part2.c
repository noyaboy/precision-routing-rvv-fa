/* GCC 14.2 RVV vsetvli pass partial-fix gap — minimal reproducer.
 *
 * Build:   riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -c minimal_part2.c
 * Inspect: riscv64-linux-objdump -d minimal_part2.o
 *
 * IMPORTANT: the bug reproduces only inside a loop. The straight-line
 * function-body form (one asm-volatile then one vzext.vf2) is correctly
 * compiled by GCC 14.2 — the optimizer's local liveness analysis emits
 * the needed vsetvli. The bug is in the *loop* version: the unified
 * vtype for the loop is hoisted to a single vsetvli OUTSIDE the loop;
 * inside the loop the asm-volatile's vsetvli changes vtype, and no
 * fresh vsetvli is emitted before the subsequent widening intrinsic.
 *
 * See paper/gcc_bug_report.md (Part 2) for full disassembly. Workaround
 * is a hand-emitted asm volatile ("vsetvli ...") before the widen.
 */
#include <riscv_vector.h>
#include <stdint.h>

#define SATURN_VFCONV_M1(dst, src, vl) do {                            \
    asm volatile (                                                     \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                          \
      "vle16.v v0, (%1)\n\t"                                           \
      ".4byte 0x4E049057\n\t"                                          \
      "vse16.v v0, (%0)\n\t"                                           \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                         \
      : "memory", "v0");                                               \
} while (0)

void widen_loop(const uint16_t *src_packed,
                uint16_t       *bf16_stash,
                float          *dst_f32,
                size_t          n_iters,
                size_t          vl) {
    for (size_t i = 0; i < n_iters; i++) {
        SATURN_VFCONV_M1(bf16_stash, src_packed + i*16, vl);
        /* WITHOUT a hand-emitted asm-volatile vsetvli e32 m2 here,
         * GCC 14.2 reuses the loop-hoisted vtype which gets clobbered
         * by the asm-volatile's inner vsetvli e16 m1 above. */
        vuint16m1_t a    = __riscv_vle16_v_u16m1(bf16_stash, vl);
        vuint32m2_t aw   = __riscv_vzext_vf2_u32m2(a, vl);
        vuint32m2_t shl  = __riscv_vsll_vx_u32m2(aw, 16, vl);
        vfloat32m2_t bf  = __riscv_vreinterpret_v_u32m2_f32m2(shl);
        __riscv_vse32_v_f32m2(dst_f32 + i*16, bf, vl);
    }
}

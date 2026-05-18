/* GCC 13.x RVV vsetvli pass miscompile — minimal reproducer.
 *
 * Build:   riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -c minimal.c
 * Inspect: riscv64-linux-objdump -d minimal.o
 *
 * Buggy on GCC 13.2 at -O2: emits one vsetvli at e16, m1 — vzext.vf2
 * then reads source EEW=8 instead of EEW=16. Silent wrong values.
 *
 * Correct on GCC 13.2 -O0, GCC 14.2 / 15.1 at any -O level: picks
 * e32, m2 as the unified vtype; vzext.vf2 reads source EEW=16.
 *
 * See paper/gcc_bug_report.md (Part 1) for full disassembly and
 * cross-version status. Workaround: asm volatile ("" : "+r"(vl));
 * between the explicit vsetvl call and the vle16.
 */
#include <riscv_vector.h>
#include <stdint.h>

void widen(const uint16_t *src, uint32_t *dst, size_t n) {
    size_t vl = __riscv_vsetvl_e32m2(n);
    vuint16m1_t a = __riscv_vle16_v_u16m1(src, vl);
    vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
    __riscv_vse32_v_u32m2(dst, b, vl);
}

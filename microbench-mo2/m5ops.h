#ifndef M5OPS_H
#define M5OPS_H

#include <stdint.h>

/*
 * gem5 RISC-V pseudo-op encoding: .long 0x0000007B | (func << 25).
 * Args go in a0 (delay), a1 (period). Both 0 = one-shot, immediate.
 * Source: gem5/util/m5/src/abi/riscv/m5op.S.
 *
 * These instructions are illegal on real hardware — gem5-only.
 */

#define M5_RESET_STATS_INSN       ".long 0x8000007B"  /* func 0x40 << 25 | 0x7B */
#define M5_DUMP_STATS_INSN        ".long 0x8200007B"  /* func 0x41 << 25 | 0x7B */
#define M5_DUMP_RESET_STATS_INSN  ".long 0x8400007B"  /* func 0x42 << 25 | 0x7B */
#define M5_EXIT_INSN              ".long 0x4200007B"  /* func 0x21 << 25 | 0x7B */
#define M5_WORK_BEGIN_INSN        ".long 0xB400007B"  /* func 0x5a << 25 | 0x7B */
#define M5_WORK_END_INSN          ".long 0xB600007B"  /* func 0x5b << 25 | 0x7B */

static inline void m5_reset_stats(uint64_t delay, uint64_t period) {
    register uint64_t a0 asm("a0") = delay;
    register uint64_t a1 asm("a1") = period;
    asm volatile (M5_RESET_STATS_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline void m5_dump_stats(uint64_t delay, uint64_t period) {
    register uint64_t a0 asm("a0") = delay;
    register uint64_t a1 asm("a1") = period;
    asm volatile (M5_DUMP_STATS_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline void m5_dump_reset_stats(uint64_t delay, uint64_t period) {
    register uint64_t a0 asm("a0") = delay;
    register uint64_t a1 asm("a1") = period;
    asm volatile (M5_DUMP_RESET_STATS_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline void m5_exit(uint64_t delay) {
    register uint64_t a0 asm("a0") = delay;
    register uint64_t a1 asm("a1") = 0;
    asm volatile (M5_EXIT_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline void m5_work_begin(uint64_t workid, uint64_t threadid) {
    register uint64_t a0 asm("a0") = workid;
    register uint64_t a1 asm("a1") = threadid;
    asm volatile (M5_WORK_BEGIN_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline void m5_work_end(uint64_t workid, uint64_t threadid) {
    register uint64_t a0 asm("a0") = workid;
    register uint64_t a1 asm("a1") = threadid;
    asm volatile (M5_WORK_END_INSN
                  : "+r"(a0), "+r"(a1)
                  :
                  : "memory");
}

static inline uint64_t rdcycle(void) {
    uint64_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
}

#endif /* M5OPS_H */

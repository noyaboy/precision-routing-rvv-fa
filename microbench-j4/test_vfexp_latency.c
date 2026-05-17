/* Track J-4 — measure vfexp.v latency on O3 to verify FU op-class wiring.
 *
 * Tight loop: vle32 → vfexp.v → vse32 (loop-carried via memory).  With
 * opLat=10 wired in FuncUnitConfig.py, each vfexp.v should take ~10 cycles
 * to retire.  Total per-iter ~ load-to-store latency through vfexp.
 *
 * Expected: with the new opLat=10 wiring, per-iter cycles >> 1.
 * Without the wiring (placeholder opLat=1), per-iter cycles ~ a few.
 */
#include <stdio.h>
#include <stdint.h>

static inline uint64_t rdcycle(void) {
    uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c;
}

int main(void) {
    enum { N = 1000 };
    static float buf[8] __attribute__((aligned(64))) = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f
    };

    asm volatile("vsetivli x0, 8, e32, m1, ta, ma" ::: );

    uint64_t t0 = rdcycle();
    for (int i = 0; i < N; i++) {
        asm volatile (
            "vle32.v  v0, (%[in])\n\t"
            ".4byte   0x4E031057\n\t"   /* vfexp.v v0, v0 */
            "vse32.v  v0, (%[in])\n\t"
            :
            : [in]"r"(buf)
            : "memory"
        );
    }
    uint64_t t1 = rdcycle();

    /* Wide-vector variant: vl=8 elts per call, so per-call cycle count
     * isolates the vfexp latency from element parallelism. */
    printf("N=%d  total_cycles=%lu  avg=%.2f cyc/iter\n",
           N, (unsigned long)(t1 - t0), (double)(t1 - t0) / N);
    printf("buf[0] sanity = %f (expected ~%f)\n",
           buf[0], 0x1.1234p+0);
    return 0;
}

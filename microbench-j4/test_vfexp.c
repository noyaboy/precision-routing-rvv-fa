/* Track J-4 POC — test the custom vfexp.v instruction added to gem5's
 * RISC-V decoder.  Encoding: VFUNCT6=0x13, VS1=0x06 (in OPFVV space).
 * For vfexp.v v0, v0: .4byte 0x4E031057.
 *
 * Issue: load FP32 input into v0, run vfexp.v, store back, compare to
 * scalar libm expf.  Pass = all elements within 1e-5 relative error
 * (libm and our semantic both use libm expf, so they should match
 * bit-exact).
 */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    float input[8]  = {0.0f, 1.0f, 2.0f, 0.69314718f, -1.0f, 0.5f, 3.0f, -2.5f};
    float output[8] = {0};

    asm volatile (
        "vsetivli x0, 8, e32, m1, ta, ma\n\t"
        "vle32.v  v0, (%[in])\n\t"
        ".4byte   0x4E031057\n\t"     /* vfexp.v v0, v0 */
        "vse32.v  v0, (%[out])\n\t"
        :
        : [in]"r"(input), [out]"r"(output)
        : "memory", "v0"
    );

    int fail = 0;
    for (int i = 0; i < 8; i++) {
        float expected = expf(input[i]);
        float diff = fabsf(output[i] - expected);
        float rel  = diff / fabsf(expected);
        printf("  vfexp(%9.6f) = %12.6e   expected %12.6e   rel_err %g\n",
               input[i], output[i], expected, rel);
        if (rel > 1e-5) fail++;
    }
    printf("\n%s (%d / 8 elements out of tolerance)\n",
           fail ? "FAIL" : "PASS", fail);
    return fail;
}

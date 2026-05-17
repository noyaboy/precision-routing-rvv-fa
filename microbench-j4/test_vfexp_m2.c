/* Track J-4 — verify vfexp.v at LMUL=2 (16 FP32 elements per vector op).
 * test_vfexp.c uses LMUL=1; the FA kernel uses LMUL=2 in the softmax loop.
 * If this passes but the FA NaN is in the vfexp asm path, the bug isn't
 * in the LMUL=2 vfexp decoder semantic.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

int main(void) {
    float input[16]  = { 0.0f, 1.0f, 2.0f, 0.69f, -1.0f, 0.5f, 3.0f, -2.5f,
                         0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f};
    float output[16] = {0};

    asm volatile (
        "vsetivli x0, 16, e32, m2, ta, ma\n\t"
        "vle32.v  v8, (%[in])\n\t"
        ".4byte   0x4E831457\n\t"     /* vfexp.v v8, v8 */
        "vse32.v  v8, (%[out])\n\t"
        :
        : [in]"r"(input), [out]"r"(output)
        : "memory", "v8", "v9"
    );

    int fail = 0;
    for (int i = 0; i < 16; i++) {
        float expected = expf(input[i]);
        float rel = fabsf(output[i] - expected) / fabsf(expected);
        if (rel > 1e-5) { printf("FAIL i=%d in=%g out=%g expected=%g\n",
                                   i, input[i], output[i], expected); fail++; }
    }
    printf("\n%s (%d / 16 out of tolerance)\n", fail ? "FAIL" : "PASS", fail);
    return fail;
}

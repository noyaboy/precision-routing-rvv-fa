/* Track J-4 POC #4 — vfconv.nvfp4.bf16.v
 * Encoding: VFUNCT6=0x13, VS1=0x09.  vfconv.nvfp4.bf16.v v0,v0 = 0x4E049057.
 * Per-element: low 4 bits of u16 = NVFP4 nibble -> BF16 output (unscaled).
 * No E4M3 scale; SW handles the scale multiply.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, sizeof(f)); return f;
}

int main(void) {
    /* Test all 16 nibbles 0..F: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0..-6 */
    uint16_t input[16];
    for (int i = 0; i < 16; i++) input[i] = (uint16_t)i;
    uint16_t output[16] = {0};

    float expected[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

    asm volatile (
        "vsetivli x0, 16, e16, m1, ta, ma\n\t"
        "vle16.v  v0, (%[in])\n\t"
        ".4byte   0x4E049057\n\t"     /* vfconv.nvfp4.bf16.v v0, v0 */
        "vse16.v  v0, (%[out])\n\t"
        :
        : [in]"r"(input), [out]"r"(output)
        : "memory", "v0"
    );

    int fail = 0;
    for (int i = 0; i < 16; i++) {
        float got = bf16_to_fp32(output[i]);
        printf("  nvfp4=0x%x -> bf16=0x%04x = %5.2f   (expected %5.2f)\n",
               i, output[i], got, expected[i]);
        if (got != expected[i]) fail++;
    }
    printf("\n%s (%d / 16 mismatches)\n", fail ? "FAIL" : "PASS", fail);
    return fail;
}

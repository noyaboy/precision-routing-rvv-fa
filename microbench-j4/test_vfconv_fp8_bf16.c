/* Track J-4 POC #2 — vfconv.fp8.bf16.v
 * Encoding: VFUNCT6=0x13, VS1=0x07.  vfconv.fp8.bf16.v v0, v0 = 0x4E039057.
 * Per-element: 8-bit FP8-E4M3 input (low byte of u16) -> BF16 output.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, sizeof(f)); return f;
}

int main(void) {
    /* FP8-E4M3 codepoints in low byte: 1.0, 2.0, 4.0, 0.5, -0.5, -1.0, 0.0, 448 */
    uint16_t input[8]  = {0x38, 0x40, 0x48, 0x30, 0xB0, 0xB8, 0x00, 0x7E};
    uint16_t output[8] = {0};

    asm volatile (
        "vsetivli x0, 8, e16, m1, ta, ma\n\t"
        "vle16.v  v0, (%[in])\n\t"
        ".4byte   0x4E039057\n\t"     /* vfconv.fp8.bf16.v v0, v0 */
        "vse16.v  v0, (%[out])\n\t"
        :
        : [in]"r"(input), [out]"r"(output)
        : "memory", "v0"
    );

    float expected[8] = {1.0f, 2.0f, 4.0f, 0.5f, -0.5f, -1.0f, 0.0f, 448.0f};
    int fail = 0;
    for (int i = 0; i < 8; i++) {
        float got = bf16_to_fp32(output[i]);
        printf("  fp8=0x%02x -> bf16=0x%04x = %g   (expected %g)\n",
               (unsigned)(input[i] & 0xff), output[i], got, expected[i]);
        if (got != expected[i]) fail++;
    }
    printf("\n%s (%d / 8 mismatches)\n", fail ? "FAIL" : "PASS", fail);
    return fail;
}

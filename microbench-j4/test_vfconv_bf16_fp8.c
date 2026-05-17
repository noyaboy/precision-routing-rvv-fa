/* Track J-4 POC #3 — vfconv.bf16.fp8.v
 * Encoding: VFUNCT6=0x13, VS1=0x08.  vfconv.bf16.fp8.v v0,v0 = 0x4E041057.
 * Per-element: BF16 input -> FP8-E4M3 byte in low byte of u16 output.
 * Algorithm matches bench_fa_common.h's bf16_to_e4m3, which is
 * bit-exact against VFConvBf16Fp8Lane.scala (0/65536 on Track F2).
 */
#include <stdio.h>
#include <stdint.h>

int main(void) {
    /* BF16 values: 1.0, 2.0, 0.5, -1.0, 448, 0.0, 4.0, 256 */
    uint16_t input[8]  = {0x3F80, 0x4000, 0x3F00, 0xBF80,
                          0x43E0, 0x0000, 0x4080, 0x4380};
    /* Expected FP8 in low byte: 0x38, 0x40, 0x30, 0xB8,
                                 0x7E, 0x00, 0x48, 0x78 */
    uint16_t output[8] = {0};
    uint8_t expected[8] = {0x38, 0x40, 0x30, 0xB8,
                           0x7E, 0x00, 0x48, 0x78};

    asm volatile (
        "vsetivli x0, 8, e16, m1, ta, ma\n\t"
        "vle16.v  v0, (%[in])\n\t"
        ".4byte   0x4E041057\n\t"     /* vfconv.bf16.fp8.v v0, v0 */
        "vse16.v  v0, (%[out])\n\t"
        :
        : [in]"r"(input), [out]"r"(output)
        : "memory", "v0"
    );

    int fail = 0;
    for (int i = 0; i < 8; i++) {
        uint8_t got = (uint8_t)(output[i] & 0xff);
        printf("  bf16=0x%04x -> fp8=0x%02x   (expected 0x%02x)\n",
               input[i], got, expected[i]);
        if (got != expected[i]) fail++;
    }
    printf("\n%s (%d / 8 mismatches)\n", fail ? "FAIL" : "PASS", fail);
    return fail;
}

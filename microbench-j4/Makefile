# Track J-4 — per-instruction functional + latency tests for the 4 Saturn
# customs against the gem5 decoder patch in ../gem5/.
#
# Build (Bootlin GCC 14.2 or compatible riscv64-linux-gnu toolchain):
#   PATH=/path/to/bootlin-riscv64/bin:$PATH make
# Outputs: 6 static rv64gcv ELFs.

CC      := riscv64-linux-gcc
ARCH    ?= rv64gcv
ABI     ?= lp64d
CFLAGS  ?= -O2 -march=$(ARCH) -mabi=$(ABI) -Wall -static -Wno-unused-result

TESTS := test_vfexp test_vfexp_m2 test_vfexp_latency \
         test_vfconv_fp8_bf16 test_vfconv_bf16_fp8 test_vfconv_nvfp4_bf16

all: $(TESTS)

test_vfexp: test_vfexp.c
	$(CC) $(CFLAGS) $< -o $@ -lm

test_vfexp_m2: test_vfexp_m2.c
	$(CC) $(CFLAGS) $< -o $@ -lm

test_vfexp_latency: test_vfexp_latency.c
	$(CC) $(CFLAGS) $< -o $@ -lm

test_vfconv_fp8_bf16: test_vfconv_fp8_bf16.c
	$(CC) $(CFLAGS) $< -o $@ -lm

test_vfconv_bf16_fp8: test_vfconv_bf16_fp8.c
	$(CC) $(CFLAGS) $< -o $@ -lm

test_vfconv_nvfp4_bf16: test_vfconv_nvfp4_bf16.c
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -f $(TESTS)

.PHONY: all clean

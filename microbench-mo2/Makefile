# Mo 2 microbench — NVFP4 K-cache vs BF16 K-cache bandwidth on gem5 SE.
# Track H adds bench_nvfp4_stub (FU-latency-stubbed dequant).
#
# Build (Bootlin GCC 13.2 or 14.2+ riscv64-linux-gnu toolchain):
#   PATH=/path/to/bootlin-riscv64/bin:$PATH make
# Outputs: ./bench_fp16, ./bench_nvfp4, ./bench_nvfp4_stub{,2}
# (static rv64gcv ELFs).

CC      := riscv64-linux-gcc
ARCH    ?= rv64gcv
ABI     ?= lp64d
CFLAGS  ?= -O2 -march=$(ARCH) -mabi=$(ABI) -Wall -static -Wno-unused-result

BENCHES := bench_fp16 bench_nvfp4 bench_nvfp4_stub bench_nvfp4_stub2

all: $(BENCHES)

bench_fp16: bench_fp16.c bench_common.h m5ops.h
	$(CC) $(CFLAGS) bench_fp16.c -o $@

bench_nvfp4: bench_nvfp4.c bench_common.h m5ops.h
	$(CC) $(CFLAGS) bench_nvfp4.c -o $@

bench_nvfp4_stub: bench_nvfp4_stub.c bench_common.h m5ops.h
	$(CC) $(CFLAGS) bench_nvfp4_stub.c -o $@

bench_nvfp4_stub2: bench_nvfp4_stub2.c bench_common.h m5ops.h
	$(CC) $(CFLAGS) bench_nvfp4_stub2.c -o $@

clean:
	rm -f $(BENCHES)

.PHONY: all clean

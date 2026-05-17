# Track J — FA decode-step microbench (Mo 5/6 prep).
#
# Run: PATH=/home/noah/project/riscv/install/bootlin-riscv64/bin:$PATH make
# Result: ./bench_fa_bf16, ./bench_fa_mixed (static rv64gcv ELFs).

CC      := riscv64-linux-gcc
ARCH    ?= rv64gcv
ABI     ?= lp64d
CFLAGS  ?= -O2 -march=$(ARCH) -mabi=$(ABI) -Wall -static -Wno-unused-result \
           -I../microbench-mo2

BENCHES := bench_fa_bf16 bench_fa_bf16_rvv bench_fa_mixed bench_fa_mixed_stub bench_fa_mixed_stub2 bench_fa_mixed_rvv bench_fa_mixed_rvv_stub bench_fa_spacemit_bf16 bench_fa_mixed_streaming_stub bench_fa_mixed_rvv_native

all: $(BENCHES)

bench_fa_bf16: bench_fa_bf16.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_bf16.c -o $@ -lm

bench_fa_bf16_rvv: bench_fa_bf16_rvv.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_bf16_rvv.c -o $@ -lm

bench_fa_mixed: bench_fa_mixed.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed.c -o $@ -lm

bench_fa_mixed_stub: bench_fa_mixed_stub.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_stub.c -o $@ -lm

bench_fa_mixed_stub2: bench_fa_mixed_stub2.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_stub2.c -o $@ -lm

bench_fa_mixed_rvv: bench_fa_mixed_rvv.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv.c -o $@ -lm

bench_fa_mixed_rvv_stub: bench_fa_mixed_rvv_stub.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv_stub.c -o $@ -lm

bench_fa_spacemit_bf16: bench_fa_spacemit_bf16.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_spacemit_bf16.c -o $@ -lm

bench_fa_mixed_streaming_stub: bench_fa_mixed_streaming_stub.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_streaming_stub.c -o $@ -lm

bench_fa_mixed_rvv_native: bench_fa_mixed_rvv_native.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv_native.c -o $@ -lm

bench_fa_mixed_rvv_native_dq: bench_fa_mixed_rvv_native_dq.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv_native_dq.c -o $@ -lm

clean:
	rm -f $(BENCHES)

.PHONY: all clean

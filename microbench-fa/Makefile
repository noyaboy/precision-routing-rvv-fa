# Track J — FA decode-step microbench (Mo 5/6 prep).
#
# Project default toolchain: Bootlin GCC 14.2 (bleeding-edge-2024.05-1)
# at /tmp/bootlin-14/. -fno-tree-vectorize is REQUIRED for the native
# bench under GCC 14 -O2 (see paper/gcc_bug_report.md). The flag is a
# no-op (<2 % cycle delta) on the all-intrinsic baselines, so we apply
# it uniformly for build consistency.
#
# Run: PATH=/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin:$PATH make
#
# To rebuild on the original GCC 13.2 (with workaround already in
# place + auto-vec on), override: PATH=/home/noah/project/riscv/install/bootlin-riscv64/bin:$PATH make CFLAGS="-O2 -march=rv64gcv -mabi=lp64d -Wall -static -Wno-unused-result -I../microbench-mo2"

CC      := riscv64-linux-gcc
ARCH    ?= rv64gcv
ABI     ?= lp64d
CFLAGS  ?= -O2 -fno-tree-vectorize -march=$(ARCH) -mabi=$(ABI) -Wall -static -Wno-unused-result \
           -I../microbench-mo2

BENCHES := bench_fa_bf16 bench_fa_bf16_rvv bench_fa_mixed bench_fa_mixed_stub bench_fa_mixed_stub2 bench_fa_mixed_rvv bench_fa_mixed_rvv_stub bench_fa_spacemit_bf16 bench_fa_mixed_streaming_stub bench_fa_mixed_rvv_native bench_fa_mixed_rvv_native_allfu

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

bench_fa_mixed_rvv_native_allfu: bench_fa_mixed_rvv_native_allfu.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv_native_allfu.c -o $@ -lm

bench_fa_mixed_rvv_native_dq: bench_fa_mixed_rvv_native_dq.c bench_fa_common.h ../microbench-mo2/m5ops.h
	$(CC) $(CFLAGS) bench_fa_mixed_rvv_native_dq.c -o $@ -lm

clean:
	rm -f $(BENCHES)

.PHONY: all clean

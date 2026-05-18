# Mo 8 step 4c — Exo-generated FA kernel microbench.
#
# Compiles the Exo-emitted exo_schedule_fa.c (fa_kernel_decode_naive)
# together with the harness in bench_fa_exo.c. Mirrors the
# microbench-fa/ build flags (Bootlin GCC 14.2, -fno-tree-vectorize)
# so cycle deltas are apples-to-apples with bench_fa_mixed_rvv_native_g14_*.
#
# Build:
#   PATH=/path/to/bootlin-riscv64-gcc14/bin:$PATH make
#
# Variants matching the bench's SEQ_LEN sweep:
#   make                    # default SEQ_LEN=2048 (L2K)
#   make L4K                # SEQ_LEN=4096
#   make L8K                # SEQ_LEN=8192
#   make L16K               # SEQ_LEN=16384

CC      := riscv64-linux-gcc
ARCH    ?= rv64gcv
ABI     ?= lp64d
CFLAGS  ?= -O2 -fno-tree-vectorize -march=$(ARCH) -mabi=$(ABI) -Wall -static \
           -Wno-unused-result -I../microbench-mo2 -I../microbench-fa

EXO_SRC := exo_schedule_fa.c
EXO_HDR := exo_schedule_fa.h
COMMON  := ../microbench-fa/bench_fa_common.h ../microbench-mo2/m5ops.h

.PHONY: all clean
all: bench_fa_exo bench_fa_exo_l4k bench_fa_exo_l8k bench_fa_exo_l16k

bench_fa_exo: bench_fa_exo.c $(EXO_SRC) $(EXO_HDR) $(COMMON)
	$(CC) $(CFLAGS) bench_fa_exo.c $(EXO_SRC) -o $@ -lm

bench_fa_exo_l4k: bench_fa_exo.c $(EXO_SRC) $(EXO_HDR) $(COMMON)
	$(CC) $(CFLAGS) -DSEQ_LEN=4096 bench_fa_exo.c $(EXO_SRC) -o $@ -lm

bench_fa_exo_l8k: bench_fa_exo.c $(EXO_SRC) $(EXO_HDR) $(COMMON)
	$(CC) $(CFLAGS) -DSEQ_LEN=8192 bench_fa_exo.c $(EXO_SRC) -o $@ -lm

bench_fa_exo_l16k: bench_fa_exo.c $(EXO_SRC) $(EXO_HDR) $(COMMON)
	$(CC) $(CFLAGS) -DSEQ_LEN=16384 bench_fa_exo.c $(EXO_SRC) -o $@ -lm

clean:
	rm -f bench_fa_exo bench_fa_exo_l4k bench_fa_exo_l8k bench_fa_exo_l16k

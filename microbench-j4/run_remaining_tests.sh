#!/bin/bash
# Builds and runs the remaining vfconv test microbenches on gem5 SE mode.
# Expects the riscv64-linux-gnu Bootlin toolchain on PATH and the gem5
# conda env activated. See ../paper/track_j4_results.md § "POC validation"
# for the original recipe.
set -e
cd "$(dirname "$0")"

: "${GEM5_DIR:?Set GEM5_DIR to the gem5 fork checkout (e.g. export GEM5_DIR=$PWD/../gem5)}"
: "${GEM5_OPT:=${GEM5_DIR}/build/RISCV/gem5.opt}"
: "${GEM5_CFG:=${GEM5_DIR}/configs/deprecated/example/se.py}"

for t in test_vfconv_bf16_fp8 test_vfconv_nvfp4_bf16; do
    echo "=== Building $t ==="
    riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -static "${t}.c" -o "${t}" -lm
    mkdir -p "run_${t}"
    (
        cd "run_${t}"
        LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
          "${GEM5_OPT}" --outdir=. "${GEM5_CFG}" \
          --cpu-type=TimingSimpleCPU --num-cpus=4 --mem-size=512MB \
          -c "../${t}" 2>&1 | tail -25
    )
done

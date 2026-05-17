#!/bin/bash
set -e
cd /home/noah/project/riscv/microbench-j4
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build
export PATH=/home/noah/project/riscv/install/bootlin-riscv64/bin:$PATH

for t in test_vfconv_bf16_fp8 test_vfconv_nvfp4_bf16; do
    echo "=== Building $t ==="
    riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -static ${t}.c -o ${t} -lm
    mkdir -p run_${t}
    cd run_${t}
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
      /home/noah/project/riscv/gem5/build/RISCV/gem5.opt --outdir=. \
      /home/noah/project/riscv/gem5/configs/deprecated/example/se.py \
      --cpu-type=TimingSimpleCPU --num-cpus=4 --mem-size=512MB \
      -c /home/noah/project/riscv/microbench-j4/${t} 2>&1 | tail -25
    cd ..
done

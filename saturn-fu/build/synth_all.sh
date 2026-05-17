#!/bin/bash
# Yosys synth of each Saturn-FU module; produces per-module stat dump.
set -e
cd "$(dirname "$0")"

MODULES=(
    "PolyExpQ2_30/PolyExpQ2_30.sv:PolyExpQ2_30"
    "VFExpLane/VFExpLane.sv VFExpLane/PolyExpQ2_30.sv:VFExpLane"
    "VFConvNvfp4Bf16Lane/VFConvNvfp4Bf16Lane.sv:VFConvNvfp4Bf16Lane"
    "VFConvBf16Fp8Lane/VFConvBf16Fp8Lane.sv:VFConvBf16Fp8Lane"
    "VFConvFp8Bf16Lane/VFConvFp8Bf16Lane.sv:VFConvFp8Bf16Lane"
)

for entry in "${MODULES[@]}"; do
    files="${entry%:*}"
    top="${entry##*:}"
    out="synth_${top}.log"

    read_cmds=""
    for f in $files; do
        read_cmds="${read_cmds}read_verilog -sv ${f}; "
    done

    echo "=================================="
    echo " Synth: ${top}"
    echo "=================================="
    yosys -p "${read_cmds} synth -top ${top}; stat" > "${out}" 2>&1
    grep -A 25 "Printing statistics" "${out}" | tail -30
    echo
done

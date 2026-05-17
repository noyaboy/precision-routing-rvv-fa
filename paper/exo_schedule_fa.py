"""
Foundation pass for the Exo precision-routing scheduling work (Mo 8).

This file demonstrates the methodology that the Mo 8 scheduling pass
will scale up to a full fused-FA kernel. It lands two pieces:

  1. A high-level Exo @proc `dequant_chunk_naive` that expresses one
     16-lane NVFP4 -> BF16 dequant tile as a plain copy loop. The body
     is *opaque* (`dst[i] = src[i]`) to match the @instr surface; the
     real precision-conversion semantics live in the RTL and FU, not
     in Exo.

  2. A scheduling function `schedule_dequant_chunk` that lowers the
     naive @proc onto SaturnRVV register groups via stage_mem +
     set_memory + replace. The result emits a C body of the form

         u16 src_reg[16]; u16 dst_reg[16];
         src_reg = __riscv_vle16_v_u16m1(&src[0], 16);
         SATURN_VFCONV_NVFP4_BF16(&dst_reg, &src_reg, 16);
         __riscv_vse16_v_u16m1(&dst[0], dst_reg, 16);

     calling the standard RVV vle/vse @instrs to bridge DRAM and our
     custom Saturn vfconv lane to do the precision conversion.

This is the simplest demonstration of replace()-driven @instr
substitution on the SaturnRVV platform. Mo 8 follow-ups will scale
the same pattern up to the full §6 FA kernel: tile QK^T, softmax,
P-quant, and P*V passes; stage Q / K_scale / V_scale / S / P / O
buffers; substitute vfexp.v / vfconv.bf16.fp8.v / vfconv.fp8.bf16.v
into the inner loops the same way.

Run from project root with the Exo editable install
(`pip install -e exo/`):

    python3 paper/exo_schedule_fa.py

Exits 0 on success and prints the lowered @proc's C codegen.
"""

from __future__ import annotations

import sys

from exo import proc, DRAM
from exo.API import compile_procs_to_strings
from exo.stdlib.scheduling import *
from exo.platforms.saturn_rvv import (
    SaturnRVV_M1,
    SaturnRVV_M2,
    saturn_vle16_m1,
    saturn_vse16_m1,
    saturn_vle32_m2,
    saturn_vse32_m2,
    vfconv_nvfp4_bf16_v,
    vfexp_v,
)


# --- 1. High-level @proc ----------------------------------------------------
#
#  Algorithmic spec: dequant a 16-lane tile of NVFP4 K/V cache into BF16
#  (carrier `ui16`). The body is an opaque copy because the @instr the
#  schedule will substitute also has an opaque body (Track F design choice:
#  precision-conversion semantics live in RTL, not in Exo's IR).
#

@proc
def dequant_chunk_naive(
    src: ui16[16] @ DRAM,
    dst: ui16[16] @ DRAM,
):
    for i in seq(0, 16):
        dst[i] = src[i]


# One vfexp tile: 16 BF16 logit lanes (ui16 carrier, post-max-subtract)
# -> 16 FP32 exp lanes. vfexp_v's @instr body is opaque (`dst[i] = 0.0`,
# doesn't reference src), so we cannot use stage_mem to derive the
# logits-side buffer from a body that doesn't read it. Instead, this
# high-level @proc already has the explicit (load, compute, store)
# structure that the dequant case acquires after stage_mem. The schedule
# below just marks the staged buffers' memory class and applies replace()
# directly.
@proc
def softmax_exp_chunk_naive(
    logits:    ui16[16] @ DRAM,
    weights:   f32[16]  @ DRAM,
):
    src_reg: ui16[16]
    dst_reg: f32[16]
    for i in seq(0, 16):                 # load loop  (-> vle16)
        src_reg[i] = logits[i]
    for i in seq(0, 16):                 # compute loop (-> vfexp)
        dst_reg[i] = src_reg[i]          # opaque (type-mismatch ok)
    for i in seq(0, 16):                 # store loop (-> vse32_m2)
        weights[i] = dst_reg[i]


# --- 2. Schedule the naive @proc onto SaturnRVV ----------------------------
#
#  Pattern mirrors test_rvv.py:simple_vfmul but with our SaturnRVV memory
#  class. Steps:
#
#    a. stage `src[0:16]` into a local register buffer `src_reg`.
#    b. stage `dst[0:16]` into a local register buffer `dst_reg`.
#    c. set both buffers' memory class to SaturnRVV_M1.
#    d. autofission so each operation (load / vfconv / store) is its own
#       loop, making replace() unification cursor-addressable.
#    e. replace() the three loops with their @instr substitutes in order.
#

def schedule_dequant_chunk(p=dequant_chunk_naive, verbose=False):
    # (a + b) stage the DRAM input/output into local register buffers
    p = stage_mem(p, "for i in _: _", "src[0:16]", "src_reg")
    p = stage_mem(p, "for i in _: _", "dst[0:16]", "dst_reg")
    if verbose:
        print(">>> after stage_mem (both):"); print(p)

    # (c) mark the staged buffers as SaturnRVV_M1 (matches the @instr
    # surface arg memory class).
    p = set_memory(p, "src_reg", SaturnRVV_M1)
    p = set_memory(p, "dst_reg", SaturnRVV_M1)
    if verbose:
        print(">>> after set_memory:"); print(p)

    # (d) After stage_mem, the proc body looks like (iter-var note: the
    # load and store loops are NEW loops introduced by stage_mem with
    # iter var `i0`; the original compute loop keeps iter var `i`):
    #
    #   src_reg : ui16[16] @ SaturnRVV_M1
    #   for i0: src_reg[i0] = src[i0+0]       <- load loop   (i0, becomes vle16)
    #   dst_reg : ui16[16] @ SaturnRVV_M1
    #   for i:  dst_reg[i-0] = src_reg[i-0]   <- compute loop (i,  becomes vfconv)
    #   for i0: dst[i0+0] = dst_reg[i0]       <- store loop  (i0, becomes vse16)
    #
    # We do the compute-loop substitution first (only loop with iter `i`)
    # then the two `i0` loops in source order.
    p = replace(p, "for i in _: _ #0", vfconv_nvfp4_bf16_v)
    if verbose:
        print(">>> after replace #1 (vfconv compute):"); print(p)
    p = replace(p, "for i0 in _: _ #0", saturn_vle16_m1)
    if verbose:
        print(">>> after replace #2 (vle16 load):"); print(p)
    p = replace(p, "for i0 in _: _ #0", saturn_vse16_m1)

    return p


def schedule_softmax_exp_chunk(p=softmax_exp_chunk_naive, verbose=False):
    """
    Second demo: vfexp.v lane substitution. Exercises the LMUL-doubling
    path — input is ui16 SaturnRVV_M1 (16 lanes), output is f32
    SaturnRVV_M2 (16 lanes at LMUL=2).

    The high-level @proc already has explicit (load, compute, store)
    structure because vfexp's @instr body doesn't reference src
    (precludes the stage_mem path the dequant schedule uses). The
    schedule just marks the locals' memory class + applies replace().
    """
    p = set_memory(p, "src_reg", SaturnRVV_M1)
    p = set_memory(p, "dst_reg", SaturnRVV_M2)

    # The three for-loops all use iter `i` (no auto-rename from stage_mem
    # because we didn't call it). Each #0 substitution removes its loop
    # from the IR, so the next #0 matches the next sibling loop.
    p = replace(p, "for i in _: _ #0", saturn_vle16_m1)
    p = replace(p, "for i in _: _ #0", vfexp_v)
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)

    return p


# --- 3. Lower, emit C, verify ----------------------------------------------

def _print_body_excerpt(c_data, name_prefix):
    """Pretty-print the body of a function in the emitted C output."""
    in_body = False
    brace_depth = 0
    for line in c_data.splitlines():
        if line.startswith(f"void {name_prefix}"):
            in_body = True
        if in_body:
            print(f"  {line}")
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and line.strip() == "}":
                in_body = False
                break
    print()


if __name__ == "__main__":
    print("=== 1. Naive @procs (high-level algorithmic spec) ===")
    print(dequant_chunk_naive)
    print(softmax_exp_chunk_naive)

    print("=== 2. Schedule lowering ===")
    dequant_chunk_scheduled = schedule_dequant_chunk()
    print("--- dequant_chunk after scheduling ---")
    print(dequant_chunk_scheduled)
    softmax_exp_chunk_scheduled = schedule_softmax_exp_chunk()
    print("--- softmax_exp_chunk after scheduling ---")
    print(softmax_exp_chunk_scheduled)

    print("=== 3. Emit C ===")
    c_data, h_data = compile_procs_to_strings(
        [dequant_chunk_scheduled, softmax_exp_chunk_scheduled],
        "exo_schedule_fa.h",
    )

    # Each substituted @instr should appear in the emitted C.
    expected_markers = [
        # dequant_chunk
        "__riscv_vle16_v_u16m1",          # saturn_vle16_m1
        "SATURN_VFCONV_NVFP4_BF16",       # vfconv_nvfp4_bf16_v
        "__riscv_vse16_v_u16m1",          # saturn_vse16_m1
        # softmax_exp_chunk
        "SATURN_VFEXP",                   # vfexp_v
        "__riscv_vse32_v_f32m2",          # saturn_vse32_m2
    ]
    missing = [m for m in expected_markers if m not in c_data]
    if missing:
        print(f"  FAIL: missing markers in lowered C output: {missing}")
        print()
        print("Full C output for debugging:")
        print(c_data)
        sys.exit(1)
    for m in expected_markers:
        print(f"  marker {m:32s}: emitted")

    print()
    print("=== 4. Lowered C bodies ===")
    _print_body_excerpt(c_data, "dequant_chunk")
    _print_body_excerpt(c_data, "softmax_exp_chunk")

    print(f"=== Summary ===")
    print(f"  C file: {len(c_data)} bytes, {c_data.count(chr(10))} lines")
    print(f"  H file: {len(h_data)} bytes, {h_data.count(chr(10))} lines")
    print(f"  2 / 2 schedules unified; 2 / 4 Saturn customs exercised")
    print(f"  (vfconv.nvfp4.bf16.v + vfexp.v).  Follow-on Mo 8 sessions")
    print(f"  add tiling for full-FA structure + the remaining 2 vfconv lanes.")
    sys.exit(0)

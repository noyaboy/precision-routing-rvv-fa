"""
Foundation pass for the Exo precision-routing scheduling work (Mo 8).

This file demonstrates the methodology that the Mo 8 scheduling pass
will scale up to a full fused-FA kernel. It lands three pieces:

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

  3. **Mo 8 step 1: head_dim=64 scaling via divide_loop.** A
     `dequant_64_naive` @proc over a 64-lane window, and a schedule
     `schedule_dequant_64` that perfect-tiles the flat loop into four
     16-element chunks via `divide_loop(p, "i", 16, ["io", "ii"],
     perfect=True)`. The per-tile staging + replace() chain from (2)
     applies inside the `io` loop. The emitted C is one outer `for
     (io = 0; io < 4; io++)` containing one `SATURN_VFCONV_NVFP4_BF16`
     call with `&src[16*io]` addressing — exactly the §6 shape for
     each head_dim chunk of QK^T and P*V.

This is the simplest demonstration of replace()-driven @instr
substitution on the SaturnRVV platform plus its first scaling step.
Mo 8 follow-ups will compose this pattern with multi-head outer
loops + softmax max/sum reductions to reach the full §6 kernel.

Run from project root with the Exo editable install
(`pip install -e exo/`):

    python3 paper/exo_schedule_fa.py

Exits 0 on success and prints the lowered @procs' C codegen.
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


# --- 2b. Scale via divide_loop: head_dim = 64 dequant ---------------------
#
#  Mo 8 step 1: take the same opaque-copy shape but over 64 lanes
#  instead of 16, then perfect-tile via divide_loop(N=16) so the
#  16-element schedule from (2) applies per outer iteration. This is
#  the smallest concrete demonstration of "scale the foundation to
#  head_dim=N via Exo tiling" — the §6 fused-FA kernel's QK^T and P*V
#  passes both have head_dim-sized inner dims that tile the same way.
#
#  64 = 16 * 4 so `perfect=True` works without a tail loop. For
#  arbitrary head_dim (96, 112, ...) the same call site switches to
#  `tail="cut"` or `tail="cut_and_guard"` to handle the remainder.
#

@proc
def dequant_64_naive(
    src: ui16[64] @ DRAM,
    dst: ui16[64] @ DRAM,
):
    for i in seq(0, 64):
        dst[i] = src[i]


def schedule_dequant_64(p=dequant_64_naive, verbose=False):
    # (a) Tile the flat 64-element loop into 4 outer chunks of 16.
    #     After this: for io in seq(0,4): for ii in seq(0,16): dst[16*io+ii] = src[16*io+ii]
    p = divide_loop(p, "i", 16, ["io", "ii"], perfect=True)
    if verbose:
        print(">>> after divide_loop:"); print(p)

    # (b) Stage the windowed views into per-tile SaturnRVV register
    #     buffers. The src_reg/dst_reg allocations land *inside* the
    #     io loop, so each io iteration has its own register group —
    #     the natural mapping to vector-register reuse per chunk.
    p = stage_mem(p, "for ii in _: _", "src[16*io:16*io+16]", "src_reg")
    p = stage_mem(p, "for ii in _: _", "dst[16*io:16*io+16]", "dst_reg")
    if verbose:
        print(">>> after stage_mem (both):"); print(p)

    # (c) Collapse the affine clutter that stage_mem leaves behind on
    #     a tiled body — buffer shapes come out as
    #     `ui16[16*io + 16 - 16*io]` and inner refs as
    #     `dst_reg[16*io + ii - 16*io]`. SaturnRVV_M1.alloc() requires
    #     a literal-decimal trailing dim (see saturn_rvv.py shape check),
    #     so the simplification has to happen before set_memory.
    p = simplify(p)
    if verbose:
        print(">>> after simplify:"); print(p)

    # (d) Mark the staged buffers as SaturnRVV_M1 register groups
    #     (matches the @instr surface memory class).
    p = set_memory(p, "src_reg", SaturnRVV_M1)
    p = set_memory(p, "dst_reg", SaturnRVV_M1)

    # (e) Iter-var note: stage_mem doesn't append "0" to the existing
    #     iter `ii`; it allocates a fresh name from the global iter
    #     pool — here `i0` for both the load and the store loops
    #     (stage_mem on the dst side reuses `i0` since the first
    #     `i0` loop's iter is no longer in scope when the second
    #     stage_mem runs). The original compute loop keeps `ii`.
    #
    #     After (b)+(c) the io-body looks like:
    #       for io in seq(0,4):
    #         src_reg: ui16[16] @ SaturnRVV_M1
    #         for i0 in seq(0,16): src_reg[i0] = src[i0 + 16*io]   <- vle16 load
    #         dst_reg: ui16[16] @ SaturnRVV_M1
    #         for ii in seq(0,16): dst_reg[ii] = src_reg[ii]        <- vfconv compute
    #         for i0 in seq(0,16): dst[i0 + 16*io] = dst_reg[i0]   <- vse16 store
    #
    #     Substitute compute first (only loop with iter `ii`), then
    #     the two `i0` siblings in source order.
    p = replace(p, "for ii in _: _ #0", vfconv_nvfp4_bf16_v)
    p = replace(p, "for i0 in _: _ #0", saturn_vle16_m1)
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
    print(dequant_64_naive)

    print("=== 2. Schedule lowering ===")
    dequant_chunk_scheduled = schedule_dequant_chunk()
    print("--- dequant_chunk after scheduling ---")
    print(dequant_chunk_scheduled)
    softmax_exp_chunk_scheduled = schedule_softmax_exp_chunk()
    print("--- softmax_exp_chunk after scheduling ---")
    print(softmax_exp_chunk_scheduled)
    dequant_64_scheduled = schedule_dequant_64()
    print("--- dequant_64 after scheduling (divide_loop + per-tile stage) ---")
    print(dequant_64_scheduled)

    print("=== 3. Emit C ===")
    c_data, h_data = compile_procs_to_strings(
        [
            dequant_chunk_scheduled,
            softmax_exp_chunk_scheduled,
            dequant_64_scheduled,
        ],
        "exo_schedule_fa.h",
    )

    # Each substituted @instr should appear in the emitted C.
    expected_markers = [
        # dequant_chunk (16-wide foundation)
        "__riscv_vle16_v_u16m1",          # saturn_vle16_m1
        "SATURN_VFCONV_NVFP4_BF16",       # vfconv_nvfp4_bf16_v
        "__riscv_vse16_v_u16m1",          # saturn_vse16_m1
        # softmax_exp_chunk
        "SATURN_VFEXP",                   # vfexp_v
        "__riscv_vse32_v_f32m2",          # saturn_vse32_m2
        # dequant_64 (Mo 8 step 1: head_dim=64 scaling via divide_loop)
        "io = 0; io < 4",                 # outer-tile loop bound
        "16 * io",                        # windowed addressing
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
    _print_body_excerpt(c_data, "dequant_64")

    print(f"=== Summary ===")
    print(f"  C file: {len(c_data)} bytes, {c_data.count(chr(10))} lines")
    print(f"  H file: {len(h_data)} bytes, {h_data.count(chr(10))} lines")
    print(f"  3 / 3 schedules unified; 2 / 4 Saturn customs exercised")
    print(f"  (vfconv.nvfp4.bf16.v + vfexp.v).  Mo 8 step 1 done: head_dim=64")
    print(f"  scales via divide_loop(perfect=True) + per-tile stage_mem.")
    print(f"  Follow-on steps add multi-head outer loops, softmax reductions,")
    print(f"  P*V pass with vfconv.bf16.fp8.v + vfconv.fp8.bf16.v.")
    sys.exit(0)

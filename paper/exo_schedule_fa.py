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
    saturn_bf16_widen_f32_m2,        # Mo 8 step 2b: BF16->FP32 widen (vzext+vsll)
    saturn_vfmacc_vv_f32m2,          # Mo 8 step 2b: FP32 vector vfmacc
    saturn_vfsub_vf_f32m2,           # Mo 8 step 2c: vec - DRAM-scalar broadcast
    saturn_f32_narrow_bf16_m2,       # Mo 8 step 2c: FP32 -> BF16 truncating narrow
    saturn_vfredmax_to_dram_f32m2,   # Mo 8 step 2c: max-reduce to DRAM scalar
    saturn_vfredusum_to_dram_f32m2,  # Mo 8 step 2c: sum-reduce to DRAM scalar
)
from exo.libs.externs import fmaxf


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


# --- 2c. §6 outer-loop shape: heads × seq_len × head_dim tile -------------
#
#  Mo 8 step 2a: wrap step 1's per-row tile schedule in the §6 fused-FA
#  outer-loop structure. The §6 kernel does K-dequant inside its QK^T
#  tile loop, but the outer-loop *shape* (heads → seq_len → head_dim tile)
#  is the same whether the body is a plain copy (here) or the full
#  QK^T-then-softmax compute (subsequent step-2 substeps 2b/2c). This
#  demo shows the methodology composes to that nested-loop shape with
#  no new @instrs.
#
#  Concrete dims: H = 8 heads, head_dim = 64. seq_len is a `size`
#  parameter so the same scheduled @proc handles L2K through L16K
#  benchmark sweeps without recompilation.
#

@proc
def fa_dequant_per_row_naive(
    seq_len: size,
    K_nvfp4: ui16[8, seq_len, 64] @ DRAM,
    K_bf16:  ui16[8, seq_len, 64] @ DRAM,
):
    assert seq_len > 0
    for h in seq(0, 8):
        for s in seq(0, seq_len):
            for i in seq(0, 64):
                K_bf16[h, s, i] = K_nvfp4[h, s, i]


def schedule_fa_dequant_per_row(p=fa_dequant_per_row_naive, verbose=False):
    # (a) Tile the innermost head_dim loop into 4 chunks of 16.
    p = divide_loop(p, "i", 16, ["io", "ii"], perfect=True)
    if verbose:
        print(">>> after divide_loop:"); print(p)

    # (b) Stage the windowed views of K_nvfp4 / K_bf16 into per-tile
    #     SaturnRVV register buffers. The window expression is 3-D
    #     because the DRAM-side buffers are [H, seq_len, head_dim]; only
    #     the trailing dim is sliced, h and s are bound from outer loops.
    p = stage_mem(p, "for ii in _: _",
                  "K_nvfp4[h, s, 16*io:16*io+16]", "src_reg")
    p = stage_mem(p, "for ii in _: _",
                  "K_bf16[h, s, 16*io:16*io+16]",  "dst_reg")
    if verbose:
        print(">>> after stage_mem:"); print(p)

    # (c) Collapse the same affine clutter step 1 surfaced — the
    #     trailing-dim shapes come out as `[16*io + 16 - 16*io]`.
    p = simplify(p)
    if verbose:
        print(">>> after simplify:"); print(p)

    # (d) Mark the staged buffers as SaturnRVV register groups.
    p = set_memory(p, "src_reg", SaturnRVV_M1)
    p = set_memory(p, "dst_reg", SaturnRVV_M1)

    # (e) Same replace() chain as step 1 (compute uses iter `ii`;
    #     staged load/store loops use iter `i0` from stage_mem's
    #     fresh-iter pool — see [[feedback-exo-scheduling-idioms]]).
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


# --- 2d. Mo 8 step 2b: QK^T inner-product chunk with dequant chain --------
#
#  First demo that adds real compute to the methodology — not just opaque
#  copy + precision conversion. One 16-lane head_dim chunk of the §6
#  QK^T tile loop body:
#
#      S_acc[i] += Q[i] * dequant_to_fp32(K_nvfp4[i])     for i in 0..16
#
#  The dequant_to_fp32(...) is a two-step chain: vfconv.nvfp4.bf16.v to
#  BF16 (ui16 carrier) at LMUL=1, then SATURN_BF16_WIDEN_F32 (vzext.vf2
#  + vsll.vi 16) to FP32 at LMUL=2. The widen path matches
#  bench_fa_mixed_rvv_native.c since gem5 lacks Zvfbfwma /
#  vfwcvtbf16.f.f.v.
#
#  Q is assumed pre-widened to FP32 outside this chunk (matches the §6
#  hand-coded approach where Q is widened once at init and reused
#  across the entire seq_len QK^T pass). S_acc is read-modify-write so
#  multiple head_dim chunks can accumulate into the same vector
#  (in step-2-follow-ons that compose this chunk over head_dim tiles).
#
#  The body intentionally produces 16 per-lane partial sums rather than
#  a single inner-product scalar. Folding 16 lanes into one S[i, j]
#  cell of the score matrix needs a vfredsum reduction; that lands in
#  step 2c (online softmax), which uses the same reduction primitive
#  for the softmax max and sum scans.
#
#  Five buffers staged onto SaturnRVV register groups:
#    - Q_reg       (f32, M2)  loaded via vle32_m2
#    - K_nvfp4_reg (ui16, M1) loaded via vle16_m1
#    - K_bf16_reg  (ui16, M1) vfconv.nvfp4.bf16.v dst
#    - K_fp32_reg  (f32, M2)  saturn_bf16_widen_f32_m2 dst
#    - S_reg       (f32, M2)  loaded via vle32_m2, vfmacc r-m-w, stored via vse32_m2
#

@proc
def qkt_chunk_naive(
    Q_fp32:  f32[16]  @ DRAM,
    K_nvfp4: ui16[16] @ DRAM,
    S_acc:   f32[16]  @ DRAM,
):
    Q_reg:       f32[16]
    K_nvfp4_reg: ui16[16]
    K_bf16_reg:  ui16[16]
    K_fp32_reg:  f32[16]
    S_reg:       f32[16]
    for i in seq(0, 16):                  # vle32_m2 (Q load)
        Q_reg[i] = Q_fp32[i]
    for i in seq(0, 16):                  # vle16_m1 (K load)
        K_nvfp4_reg[i] = K_nvfp4[i]
    for i in seq(0, 16):                  # vfconv.nvfp4.bf16.v
        K_bf16_reg[i] = K_nvfp4_reg[i]
    for i in seq(0, 16):                  # SATURN_BF16_WIDEN_F32
        K_fp32_reg[i] = K_bf16_reg[i]
    for i in seq(0, 16):                  # vle32_m2 (S_acc load)
        S_reg[i] = S_acc[i]
    for i in seq(0, 16):                  # vfmacc.vv on f32m2
        S_reg[i] += Q_reg[i] * K_fp32_reg[i]
    for i in seq(0, 16):                  # vse32_m2 (S_acc store)
        S_acc[i] = S_reg[i]


def schedule_qkt_chunk(p=qkt_chunk_naive, verbose=False):
    """
    Same structure as schedule_softmax_exp_chunk — the @proc already
    carries the explicit (load, ..., compute, ..., store) sequence so
    the schedule just sets the staged buffers' memory class and applies
    replace() in source order. No stage_mem needed (vfmacc's body is
    `dst[i] += lhs[i] * rhs[i]`, real arithmetic, but Exo's unification
    matches it just like the opaque copies).
    """
    p = set_memory(p, "Q_reg",       SaturnRVV_M2)
    p = set_memory(p, "K_nvfp4_reg", SaturnRVV_M1)
    p = set_memory(p, "K_bf16_reg",  SaturnRVV_M1)
    p = set_memory(p, "K_fp32_reg",  SaturnRVV_M2)
    p = set_memory(p, "S_reg",       SaturnRVV_M2)

    # All 7 loops use iter `i`; each replace() removes its loop, so
    # `#0` advances to the next sibling. Source order matches the
    # @instr substitution order below.
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)             # Q load
    p = replace(p, "for i in _: _ #0", saturn_vle16_m1)             # K load
    p = replace(p, "for i in _: _ #0", vfconv_nvfp4_bf16_v)         # vfconv
    p = replace(p, "for i in _: _ #0", saturn_bf16_widen_f32_m2)    # widen
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)             # S_acc load
    p = replace(p, "for i in _: _ #0", saturn_vfmacc_vv_f32m2)      # vfmacc
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)             # S_acc store
    if verbose:
        print(">>> after replace chain:"); print(p)
    return p


# --- 2e. Mo 8 step 2c: online-softmax chunk (intra-tile) -------------------
#
#  Single-tile softmax kernel: given a 16-lane FP32 score vector S
#  (from one head_dim chunk of QK^T accumulation), produce:
#    - P_fp32: 16 FP32 exp values (post max-subtract), for downstream
#      P·V matmul (step 3 wires the P·V dequant lanes).
#    - m_out:  FP32 scalar = max over S (combined with prior tile's
#      m_out if the caller initializes m_out[0] = prior max; for a
#      fresh single-tile call, initialize to -inf or a large negative).
#    - l_out:  FP32 scalar = sum over P (combined with prior tile's
#      l_out if the caller initializes l_out[0] = prior sum;
#      for fresh, initialize to 0.0).
#
#  This is the single-tile softmax; cross-tile rescaling (multiplying
#  prior O_accumulator by exp(m_prev - m_new) when m_new > m_prev)
#  is the responsibility of the §6 outer loop and lands when step 3
#  composes this chunk with the P·V pass.
#
#  Pipeline:
#    S_reg       (M2, f32) <- S_fp32                  [vle32_m2]
#    m_out[0]    (DRAM)    <- fmaxf-reduce(S_reg)    [vfredmax]
#    S_shifted   (M2, f32) <- S_reg - m_out[0]        [vfsub.vf]
#    S_bf16      (M1, ui16)<- narrow(S_shifted)       [F32_NARROW_BF16]
#    P_reg       (M2, f32) <- vfexp(S_bf16)           [vfexp_v]
#    P_fp32      (DRAM)    <- P_reg                   [vse32_m2]
#    l_out[0]    (DRAM)    <- sum-reduce(P_reg)       [vfredusum]
#
#  Note: vfexp_v consumes BF16 carriers (SEW=16/m1) since the Saturn
#  VFExpLane RTL operates on the BF16 input precision. The narrow
#  step bridges from the FP32-softmax precision (matches paper
#  precision_config) down to the exp lane's BF16 surface.
#

@proc
def online_softmax_chunk_naive(
    S_fp32: f32[16] @ DRAM,
    P_fp32: f32[16] @ DRAM,
    m_out:  f32[1]  @ DRAM,
    l_out:  f32[1]  @ DRAM,
):
    S_reg:       f32[16]
    S_shifted:   f32[16]
    S_bf16:      ui16[16]
    P_reg:       f32[16]
    for i in seq(0, 16):                       # vle32_m2 (S load)
        S_reg[i] = S_fp32[i]
    for i in seq(0, 16):                       # vfredmax -> DRAM scalar
        m_out[0] = fmaxf(m_out[0], S_reg[i])
    for i in seq(0, 16):                       # vfsub.vf (broadcast subtract)
        S_shifted[i] = S_reg[i] - m_out[0]
    for i in seq(0, 16):                       # F32_NARROW_BF16
        S_bf16[i] = S_shifted[i]
    for i in seq(0, 16):                       # vfexp_v
        P_reg[i] = S_bf16[i]
    for i in seq(0, 16):                       # vse32_m2 (P store)
        P_fp32[i] = P_reg[i]
    for i in seq(0, 16):                       # vfredusum -> DRAM scalar
        l_out[0] += P_reg[i]


def schedule_online_softmax_chunk(p=online_softmax_chunk_naive, verbose=False):
    """
    Same pattern as schedule_qkt_chunk — @proc carries explicit
    (load, ..., compute, ..., store) sequence so the schedule just
    annotates memory + replaces 7 loops in source order. Sibling
    `#0` index advances naturally as each replace() removes its
    matched loop.
    """
    p = set_memory(p, "S_reg",     SaturnRVV_M2)
    p = set_memory(p, "S_shifted", SaturnRVV_M2)
    p = set_memory(p, "S_bf16",    SaturnRVV_M1)
    p = set_memory(p, "P_reg",     SaturnRVV_M2)

    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)              # S load
    p = replace(p, "for i in _: _ #0", saturn_vfredmax_to_dram_f32m2)# max-reduce
    p = replace(p, "for i in _: _ #0", saturn_vfsub_vf_f32m2)        # S - max
    p = replace(p, "for i in _: _ #0", saturn_f32_narrow_bf16_m2)    # FP32->BF16
    p = replace(p, "for i in _: _ #0", vfexp_v)                      # exp
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)              # P store
    p = replace(p, "for i in _: _ #0", saturn_vfredusum_to_dram_f32m2) # sum-reduce
    if verbose:
        print(">>> after replace chain:"); print(p)
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
    print(fa_dequant_per_row_naive)
    print(qkt_chunk_naive)
    print(online_softmax_chunk_naive)

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
    fa_dequant_per_row_scheduled = schedule_fa_dequant_per_row()
    print("--- fa_dequant_per_row after scheduling (§6 outer-loop shape) ---")
    print(fa_dequant_per_row_scheduled)
    qkt_chunk_scheduled = schedule_qkt_chunk()
    print("--- qkt_chunk after scheduling (Mo 8 step 2b: QK^T inner-product) ---")
    print(qkt_chunk_scheduled)
    online_softmax_chunk_scheduled = schedule_online_softmax_chunk()
    print("--- online_softmax_chunk after scheduling (Mo 8 step 2c) ---")
    print(online_softmax_chunk_scheduled)

    print("=== 3. Emit C ===")
    c_data, h_data = compile_procs_to_strings(
        [
            dequant_chunk_scheduled,
            softmax_exp_chunk_scheduled,
            dequant_64_scheduled,
            fa_dequant_per_row_scheduled,
            qkt_chunk_scheduled,
            online_softmax_chunk_scheduled,
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
        # fa_dequant_per_row (Mo 8 step 2a: §6 outer-loop shape)
        "h = 0; h < 8",                   # heads outer loop
        "s = 0; s < seq_len",             # seq_len middle loop (`size` param)
        # qkt_chunk (Mo 8 step 2b: QK^T inner-product with dequant chain)
        "SATURN_BF16_WIDEN_F32",          # saturn_bf16_widen_f32_m2 macro
        "__riscv_vfmacc_vv_f32m2",        # saturn_vfmacc_vv_f32m2 intrinsic
        "__riscv_vle32_v_f32m2",          # saturn_vle32_m2 (Q + S_acc loads)
        # online_softmax_chunk (Mo 8 step 2c: intra-tile online softmax)
        "SATURN_VFREDMAX_F32M2",          # saturn_vfredmax_to_dram_f32m2 macro
        "__riscv_vfsub_vf_f32m2",         # saturn_vfsub_vf_f32m2 intrinsic
        "SATURN_F32_NARROW_BF16",         # saturn_f32_narrow_bf16_m2 macro
        "SATURN_VFREDUSUM_F32M2",         # saturn_vfredusum_to_dram_f32m2 macro
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
    _print_body_excerpt(c_data, "fa_dequant_per_row")
    _print_body_excerpt(c_data, "qkt_chunk")
    _print_body_excerpt(c_data, "online_softmax_chunk")

    print(f"=== Summary ===")
    print(f"  C file: {len(c_data)} bytes, {c_data.count(chr(10))} lines")
    print(f"  H file: {len(h_data)} bytes, {h_data.count(chr(10))} lines")
    print(f"  6 / 6 schedules unified; all 4 Saturn customs exercised")
    print(f"  (vfconv.nvfp4.bf16.v + vfexp.v + bf16->fp32 widen + fp32 narrow).")
    print(f"  Mo 8 steps 1 + 2a + 2b + 2c done:")
    print(f"  - step 1:  head_dim=64 via divide_loop + per-tile stage_mem")
    print(f"  - step 2a: §6 outer-loop shape (8 heads × seq_len × head_dim tile)")
    print(f"  - step 2b: QK^T inner-product chunk with dequant chain feeding")
    print(f"             fp32 vfmacc.vv")
    print(f"  - step 2c: intra-tile online softmax — max-reduce + vfsub.vf +")
    print(f"             FP32->BF16 narrow + vfexp + sum-reduce, all driven by")
    print(f"             4 new @instrs (vfredmax/vfredusum/vfsub.vf/narrow)")
    print(f"  Follow-on: step 3 wires remaining 2 vfconv lanes (bf16.fp8.v +")
    print(f"  fp8.bf16.v) for P-quant + P*V dequant.")
    sys.exit(0)

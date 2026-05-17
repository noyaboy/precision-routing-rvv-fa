"""
Smoke test for the SaturnRVV Exo platform (Track D-follow, 2026-05-17).

Confirms that:
  1. All 4 @instr declarations type-check.
  2. A trivial @proc that allocates a SaturnRVV buffer and calls one of
     the @instrs compiles to C without error.
  3. The emitted C contains the expected `SATURN_*` macro invocation
     and the inline-asm `.4byte` encodings from the platform's
     `c_global` header block.

Run from project root with the Exo editable install (`pip install -e
exo/`):

    python3 paper/exo_smoke_test.py

Exits 0 on success; prints a summary table of what was verified.
"""

from __future__ import annotations

import sys
from exo import proc, DRAM
from exo.API import compile_procs_to_strings
from exo.platforms.saturn_rvv import (
    SaturnRVV_M1,
    SaturnRVV_M2,
    SaturnRVV_M4,
    vfconv_nvfp4_bf16_v,
    vfconv_bf16_fp8_v,
    vfconv_fp8_bf16_v,
    vfexp_v,
)


# --- 1. Import / type-check sanity -----------------------------------------

print("=== 1. Imports + @instr type-check ===")
for cls in (SaturnRVV_M1, SaturnRVV_M2, SaturnRVV_M4):
    print(f"  {cls.__name__:14s}: VLEN={cls.VLEN_BITS} LMUL={cls.LMUL}")
for fn in (vfconv_nvfp4_bf16_v, vfconv_bf16_fp8_v, vfconv_fp8_bf16_v, vfexp_v):
    print(f"  @instr {fn.name():25s}: OK")


# --- 2. Trivial @proc that allocates SaturnRVV buffers + calls each --------

# Procs take SaturnRVV buffers as arguments directly — sidesteps the
# can_read=False restriction on element-wise scalar transfers. In a full
# kernel, vle / vse @instrs (a follow-on TODO in saturn_rvv.py) would
# bridge DRAM <-> SaturnRVV; here we just exercise the @instr call site.


@proc
def call_vfexp(src: [ui16][16] @ SaturnRVV_M1,
               dst: [f32][16]  @ SaturnRVV_M2,
               vl: size):
    assert vl <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    vfexp_v(dst[0:16], src[0:16], vl)


@proc
def call_nvfp4_dequant(src: [ui16][16] @ SaturnRVV_M1,
                       dst: [ui16][16] @ SaturnRVV_M1,
                       vl: size):
    assert vl <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    vfconv_nvfp4_bf16_v(dst[0:16], src[0:16], vl)


@proc
def call_bf16_to_fp8(src: [ui16][16] @ SaturnRVV_M1,
                     dst: [ui8][32]  @ SaturnRVV_M1,
                     vl: size):
    assert vl <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    vfconv_bf16_fp8_v(dst[0:32], src[0:16], vl)


@proc
def call_fp8_to_bf16(src: [ui8][32]  @ SaturnRVV_M1,
                     dst: [ui16][16] @ SaturnRVV_M1,
                     vl: size):
    assert vl <= 16
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    vfconv_fp8_bf16_v(dst[0:16], src[0:32], vl)


print()
print("=== 2. @proc construction ===")
for p in (call_vfexp, call_nvfp4_dequant, call_bf16_to_fp8, call_fp8_to_bf16):
    print(f"  @proc {p.name():22s}: OK")


# --- 3. Emit C and check for expected content ------------------------------

print()
print("=== 3. C codegen ===")
c_data, h_data = compile_procs_to_strings(
    [call_vfexp, call_nvfp4_dequant, call_bf16_to_fp8, call_fp8_to_bf16],
    "saturn_smoke.h",
)

# Check: each per-lane macro invocation appears in the C output.
expected_macros = [
    "SATURN_VFEXP",
    "SATURN_VFCONV_NVFP4_BF16",
    "SATURN_VFCONV_BF16_FP8",
    "SATURN_VFCONV_FP8_BF16",
]
missing = [m for m in expected_macros if m not in c_data]
if missing:
    print(f"  FAIL: missing macros in C output: {missing}")
    sys.exit(1)
for m in expected_macros:
    print(f"  macro call {m:28s}: emitted")

# Check: the `c_global` header block (with .4byte encodings) lands in
# the generated source.
expected_encodings = [
    "0x4E831457",  # vfexp.v v8, v8
    "0x4E049057",  # vfconv.nvfp4.bf16.v v0, v0
    "0x4E841457",  # vfconv.bf16.fp8.v v8, v8
    "0x4E839057",  # vfconv.fp8.bf16.v v0, v8
]
missing_enc = [e for e in expected_encodings if e not in c_data]
if missing_enc:
    print(f"  FAIL: missing .4byte encodings in C output: {missing_enc}")
    sys.exit(1)
for e in expected_encodings:
    print(f"  encoding   {e}: emitted")

# Check: the SaturnRVV memory class's global_() header is emitted
# (proves the platform is recognized by codegen).
expected_global = [
    "#ifndef SATURN_CUSTOM_ASM_H",
    "#include <riscv_vector.h>",
]
missing_glob = [g for g in expected_global if g not in c_data]
if missing_glob:
    print(f"  FAIL: missing c_global markers in C output: {missing_glob}")
    sys.exit(1)
for g in expected_global:
    print(f"  c_global   {g[:30]:30s}: emitted")

# Also dump just the per-@proc function bodies (skip the repeated
# c_global header blocks) so the macro call sites are visible.
print()
print("=== Emitted C (proc bodies, excerpts) ===")
in_body = False
brace_depth = 0
for i, line in enumerate(c_data.splitlines(), 1):
    if line.startswith("void call_"):
        in_body = True
        brace_depth = 0
    if in_body:
        print(f"  {i:3d}  {line}")
        brace_depth += line.count("{") - line.count("}")
        if brace_depth == 0 and "{" in c_data.splitlines()[i-1] or (
            brace_depth == 0 and line.strip() == "}"
        ):
            in_body = False
            print()

# --- Final summary ---------------------------------------------------------

print()
print(f"=== Summary ===")
print(f"  C file: {len(c_data)} bytes, {c_data.count(chr(10))} lines")
print(f"  H file: {len(h_data)} bytes, {h_data.count(chr(10))} lines")
print(f"  All checks passed.")
sys.exit(0)

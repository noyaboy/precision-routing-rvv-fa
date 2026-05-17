# Exo 2 `@instr` declarations for Saturn custom instructions (Track D, 2026-05-17)

Pins the **compiler-side interface** for the 4 Saturn precision-routing custom instructions before
more RTL or schedule work. Grounded in real Exo 2 conventions read from `exo/src/exo/API.py:52` and
the upstream `exo/src/exo/platforms/{rvv,neon,sve_vla,gemmini}.py` reference platforms.

Replaces the "syntax illustrative" placeholder in `fu_sketch.md §Exo` with declarations that
compile against current Exo 2 (commit `2f5472d`, 2026-01-08) once the type-support extension in
§3 is merged.

## 0. Why this artifact exists

- Paper framing constraint: **must** lead with compiler co-design. After Mo 2 PASS and Mo 3
  (two checkpoints) DONE, Y1 has zero compiler-side artifacts. This is the first.
- Mo 8 checkpoint asks "Exo-generated mixed-prec FA within 10% of hand-coded?" — feasibility was
  unverified until each of our 4 custom instructions had a real `@instr` body. Failure to express
  one would force Y1 replanning.
- Verdict (preview, justified below): **Exo 2 `@instr` can express all 4 instructions with a
  ~30-line type-extension patch (BF16 first-class) + opaque `uint8` carriers for FP8/NVFP4.**
  No semantic blockers found. Mo 8 risk is now scheduling/perf, not feasibility.

## 1. Reality check vs. upstream Exo 2

| Component (Y1 needs) | Upstream state (commit `2f5472d`, 2026-01-08) | Gap |
|---|---|---|
| `@instr` decorator | Yes (`src/exo/API.py:52`) — `@instr(c_instr, c_global="")` | None |
| RVV memory class | Yes (`src/exo/platforms/rvv.py:19`, class `RVV`) | f32-only, **VLEN=128 hardcoded** (vec_types maps `"float" → (4, "vfloat32m1_t")`), 9 intrinsics, no LMUL |
| BF16 primitive type | **Missing** | Frontend supports `f16`/`f32`/`f64`/`i8`/`u8`/`u16`/`i32` only (`src/exo/frontend/typecheck.py:594-604`). Need to add `bf16` |
| FP8-E4M3 primitive type | Missing | Carry as opaque `u8` (no IEEE arithmetic needed inside Exo) |
| NVFP4 (E2M1) primitive type | Missing | Carry as packed `u8` (2 nibbles/byte) |
| Per-block scale layout | Expressible (`stride()` + window) | Needs convention; spelled out in §5 |
| VLEN-parametric vsetvl | Missing | Stub `rvv.py` hardcodes `vl=4`. Need `@config` for vsetvl (gemmini.py:284 pattern) |
| LMUL parametric memory class | Missing | Need parametric vec type per `(dtype, lmul)` |
| Custom-instruction inline asm | Permissive (any C string in `@instr`) | None — Saturn customs lacking GCC intrinsics use `asm volatile(...)` |

**Memory delta:** the prior memory entry that "Exo trunk added ARM SVE May 2025, NO RVV yet" is
**wrong** — `src/exo/platforms/rvv.py` (175 LOC) is in upstream. But the file is a minimal VLEN=128
f32 stub (9 intrinsics, no LMUL, no BF16/FP8/NVFP4), not a full RVV scheduling library. The "no
RVV scheduling library / no LLM kernel coverage" framing the paper relies on still holds; the
"file doesn't exist" claim does not. Updating `reference_key_papers_repos.md` accordingly.

## 2. Exo 2 facts that matter here (verified from source, not paper)

- `@instr(c_instr, c_global="")`: minimal — takes a C-format string with `{name}` and `{name_data}`
  placeholders. The decorator wraps a regular Exo `proc` whose body provides **semantics** (used by
  the `replace()` scheduling op to verify equivalence). No first-class fields for funct6, latency,
  or vsetvl group.
- Primitive types: `T.f16, T.f32, T.f64, T.int8, T.uint8, T.uint16, T.int32, T.bool` plus `int /
  size / index` for indices (`src/exo/frontend/typecheck.py:592-606` and `src/exo/core/LoopIR.py:387-397`).
- Memory class abstraction (`Memory`): four classmethods — `global_()`, `alloc()`, `free()`,
  `window()` — plus `can_read()` flag (`RVV.can_read = False` means buffer values are opaque to Exo,
  Exo can only do bulk moves via `@instr` calls).
- `@config` (`API.py:65`): declares a stateful class for setup-style state (used by gemmini for
  config_zero/config_matmul). This is the right mechanism for `vsetvl` group state.
- Replace mechanism (`replace(p, "for i in _: _", my_instr)`): pattern-matches a loop body against
  an `@instr`'s body. If isomorphic up to assertions, substitutes the `@instr` call. This is the
  load-bearing primitive for "Exo generates a kernel that uses our custom instructions."

## 3. Required Exo extensions for Y1 (small, scoped)

The Y1 paper claim is "Exo 2 extension with `@instr`-declared precision-routing instructions."
Concrete extension surface (estimated ~150 LOC across 4 files):

1. **Add `bf16` to primitive types.** Mirror `f16` everywhere it appears: `core/LoopIR.py`
   (~10 LOC for the type class + alias), `frontend/typecheck.py` (~5 LOC for the UAST → T entry),
   `frontend/parser.py` (~5 LOC for `bf16` keyword recognition), `backend/LoopIR_compiler.py`
   (~10 LOC for `_Float16`/`__bf16` C-type mapping under `-mbf16` GCC support). ~30 LOC total.
2. **`SaturnRVV` parametric memory class** (~80 LOC, new file in `paper/exo_instr_decls.md`'s
   companion `src/exo/platforms/saturn_rvv.py` once built). Drop-in replacement for stock `RVV`,
   parameterized over `(prim_type, lmul)` to produce the right `vfloat32mN_t` / `vbfloat16mN_t` /
   `vuint8mN_t` C-type alias.
3. **`vsetvl_config`** (`@config` class) for vsetvl state — sew/lmul. Holds active VL so successive
   intrinsics share it. ~20 LOC.
4. **VLEN-agnostic `vle` / `vse`** (~20 LOC of additional `@instr` declarations alongside the
   custom 4). Reuses pattern from `rvv.py` but with `vl: size` arg from `sve_vla.py:10`.

Carriers: FP8-E4M3 → `uint8`. NVFP4 (E2M1) → `uint8` with packed-nibble convention (per-byte
unpacking inside `@instr` body, so Exo sees byte-level semantics). Block scales for NVFP4 → `uint8`
buffer with stride-16 layout (one E4M3 scale per 16 NVFP4 elements).

Adding BF16 as first-class (vs. carrying as `uint16`) buys the paper's "compiler precision-routing
pass picks per-stage dtype" claim — Exo sees the type, not opaque bytes.

## 4. Memory class declaration

Sketch (target file: `src/exo/platforms/saturn_rvv.py`). Real declaration during Mo 5–8
RTL-integration; this pins the interface today.

```python
from __future__ import annotations
from exo import Memory, DRAM, instr, config
from exo.core.memory import MemGenError

# Parametric: per-(dtype, LMUL) Saturn RVV register-class memory.
# vec_types maps Exo prim_type -> (C type, lanes_per_lmul1).
# lanes scale linearly with lmul: vbfloat16m2_t has 2*lanes_per_lmul1 lanes.

_SATURN_VLEN_BITS = 256   # gem5 target; production parameterizes via env

def _saturn_rvv_class(lmul: int):
    class _Cls(Memory):
        VLEN_BITS = _SATURN_VLEN_BITS
        LMUL = lmul

        @classmethod
        def global_(cls):
            return '#include <riscv_vector.h>\n#include "saturn_custom_asm.h"'

        @classmethod
        def can_read(cls):
            return False  # opaque except via @instr

        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            vec_types = {
                "float":     ("vfloat32",   32),  # f32
                "__bf16":    ("vbfloat16",  16),  # bf16  (requires Exo BF16 patch)
                "uint8_t":   ("vuint8",      8),  # carries FP8 / packed NVFP4
                "uint16_t":  ("vuint16",    16),
            }
            if prim_type not in vec_types:
                raise MemGenError(f"{srcinfo}: SaturnRVV unsupported {prim_type}")
            stem, elem_bits = vec_types[prim_type]
            lanes = (cls.VLEN_BITS * cls.LMUL) // elem_bits
            if shape and shape[-1].isdecimal() and int(shape[-1]) != lanes:
                raise MemGenError(
                    f"{srcinfo}: SaturnRVV<{prim_type}, m{cls.LMUL}> "
                    f"needs trailing dim == {lanes}, got {shape[-1]}")
            ctype = f"{stem}m{cls.LMUL}_t"
            return f"{ctype} {new_name};"

        @classmethod
        def free(cls, *args, **kwargs): return ""

        @classmethod
        def window(cls, basetyp, baseptr, indices, strides, srcinfo):
            assert strides[-1] == "1"
            idxs = indices[:-1] or ""
            return f"{baseptr}{('[' + ']['.join(idxs) + ']') if idxs else ''}"
    _Cls.__name__ = f"SaturnRVV_M{lmul}"
    return _Cls

SaturnRVV_M1 = _saturn_rvv_class(1)
SaturnRVV_M2 = _saturn_rvv_class(2)
```

At VLEN=256 / LMUL=1: bf16 lanes = 16, f32 lanes = 8, u8 lanes = 32. The vfexp lane (BF16 →
FP32, LMUL-doubling) emits to a `SaturnRVV_M2` f32 buffer of 16 lanes.

## 5. `@instr` declarations for the 4 Saturn custom instructions

Each follows the upstream Exo convention: C-code template with `{x_data}` substitution for
buffer pointers and `{x}` for scalar names. Body provides semantics (used by `replace()` to verify).
Saturn customs likely lack GCC intrinsics, so we route through inline-asm helper macros in a
`saturn_custom_asm.h` header (companion C file, not Exo-side).

### 5.1 `vfconv.nvfp4.bf16.v` — NVFP4 K-cache → BF16

```python
# vd (BF16, LMUL=2) <- dequant(NVFP4 mantissas in vs2, E4M3 scales in vs1).
# Per RVV: vs2 is uint8 vec carrying 2 NVFP4 nibbles per byte;
#          vs1 is uint8 vec carrying 1 E4M3 scale per 16 NVFP4 elements (block-16).

@instr(
    "SATURN_VFCONV_NVFP4_BF16("
    "&{vd_data}, &{vs2_data}, &{vs1_data}, {vl});",
    c_global='#include "saturn_custom_asm.h"',  # asm volatile wrapper
)
def vfconv_nvfp4_bf16_v(
    vl:   size,
    vd:   [bf16][32] @ SaturnRVV_M2,   # 32 BF16 outputs / call @ VLEN=256
    vs2:  [u8][16]   @ SaturnRVV_M1,   # 16 bytes = 32 packed NVFP4 mantissas
    vs1:  [u8][2]    @ SaturnRVV_M1,   # 2 E4M3 scales (one per block of 16 outs)
):
    assert stride(vd,  0) == 1
    assert stride(vs2, 0) == 1
    assert stride(vs1, 0) == 1
    assert vl >= 0
    assert vl <= 32

    # Semantics body: serves Exo's replace() equivalence check.
    # NVFP4 unpack is a 16-entry LUT; E4M3 scale unpack + multiply gives the BF16.
    # (Logical, not bit-exact — Exo only checks shape/type/effects, not numerics.)
    for i in seq(0, vl):
        # i // 16 picks the right scale; i % 2 picks high/low nibble.
        # nvfp4_lut / e4m3_to_bf16 are extern decls (see §6).
        vd[i] = nvfp4_lut(vs2[i / 2], i % 2) * e4m3_to_bf16(vs1[i / 16])
```

### 5.2 `vfconv.bf16.fp8.v` — BF16 attention weights → FP8

```python
@instr(
    "SATURN_VFCONV_BF16_FP8(&{vd_data}, &{vs2_data}, {vl});",
    c_global='#include "saturn_custom_asm.h"',
)
def vfconv_bf16_fp8_v(
    vl:   size,
    vd:   [u8][16]   @ SaturnRVV_M1,   # FP8-E4M3 outputs, packed 1B each
    vs2:  [bf16][16] @ SaturnRVV_M1,
):
    assert stride(vd,  0) == 1
    assert stride(vs2, 0) == 1
    assert vl >= 0
    assert vl <= 16

    for i in seq(0, vl):
        vd[i] = bf16_to_e4m3_rne(vs2[i])   # RNE + saturate to E4M3 range
```

### 5.3 `vfconv.fp8.bf16.v` — FP8 attention weights → BF16 (for V matmul)

```python
@instr(
    "SATURN_VFCONV_FP8_BF16(&{vd_data}, &{vs2_data}, {vl});",
    c_global='#include "saturn_custom_asm.h"',
)
def vfconv_fp8_bf16_v(
    vl:   size,
    vd:   [bf16][32] @ SaturnRVV_M2,   # LMUL-doubling: 16 fp8 in -> 32 bf16 out
    vs2:  [u8][16]   @ SaturnRVV_M1,
):
    assert stride(vd,  0) == 1
    assert stride(vs2, 0) == 1
    assert vl >= 0
    assert vl <= 16

    for i in seq(0, vl):
        vd[i] = e4m3_to_bf16(vs2[i])
```

### 5.4 `vfexp.v` — BF16 logit → FP32 inside softmax kernel

```python
@instr(
    "SATURN_VFEXP(&{vd_data}, &{vs2_data}, {vl});",
    c_global='#include "saturn_custom_asm.h"',
)
def vfexp_v(
    vl:   size,
    vd:   [f32][16]  @ SaturnRVV_M2,   # 16 FP32 outputs (LMUL-doubling vs BF16 in)
    vs2:  [bf16][16] @ SaturnRVV_M1,
):
    assert stride(vd,  0) == 1
    assert stride(vs2, 0) == 1
    assert vl >= 0
    assert vl <= 16

    for i in seq(0, vl):
        vd[i] = expf_bf16_to_f32(vs2[i])   # semantic placeholder, see §6
```

## 6. Externs

Six extern decls in `src/exo/libs/externs.py` (mirroring `relu` / `select`). Pure semantics: Exo
treats them opaquely, only requires that they exist as C functions linked at compile time.

| Extern | C signature | Used by |
|---|---|---|
| `nvfp4_lut(byte, nibble_idx)` | `static inline __bf16 nvfp4_lut(uint8_t, int);` | §5.1 |
| `e4m3_to_bf16(byte)` | `static inline __bf16 e4m3_to_bf16(uint8_t);` | §5.1, §5.3 |
| `bf16_to_e4m3_rne(bf)` | `static inline uint8_t bf16_to_e4m3_rne(__bf16);` | §5.2 |
| `expf_bf16_to_f32(bf)` | `static inline float expf_bf16_to_f32(__bf16);` | §5.4 |

Externs let Exo's `replace()` check **shape and type compatibility** without exposing the bit-level
internals (which live in Chisel — `VFExpLane.scala`, `VFConvNvfp4Bf16Lane.scala`, etc.).

## 7. `vsetvl` strategy

Saturn RVV uses `vsetvl rd, rs1, sew, lmul` to set VL/SEW/LMUL group state. Stock `rvv.py` skips
this (hardcoded `vl=4`). Pattern from `gemmini.py:284`:

```python
@config
class vsetvl_state:
    sew:  size   # 8/16/32
    lmul: size   # 1/2/4/8
    vl:   size

@instr("__riscv_vsetvl_e{sew}m{lmul}({vl_data});")
def vsetvl_e_m(sew: size, lmul: size, vl: size @ DRAM):
    ...  # body: updates the config
```

This makes vsetvl visible to Exo's scheduler so loop nests sharing a VL hoist a single `vsetvl`
out of the inner loop (the upstream stub doesn't do this).

## 8. End-to-end schedule outline (where these `@instr`s plug in)

Sketch of the FA kernel as Exo will see it (a top-level `@proc` describing the per-tile attention
math), with `replace()` calls swapping loops for our `@instr`s. This is what Mo 8 has to produce.

```python
@proc
def flash_attn_tile_mixed(
    Q:        f32[D, BLK_M] @ DRAM,
    K_nvfp4:  u8[D//2, BLK_N] @ DRAM,
    K_scales: u8[D//16, BLK_N] @ DRAM,
    V_nvfp4:  u8[D//2, BLK_N] @ DRAM,
    V_scales: u8[D//16, BLK_N] @ DRAM,
    O:        f32[D, BLK_M] @ DRAM,
):
    # 1) Load K block, dequant to BF16    -> uses vfconv.nvfp4.bf16.v
    # 2) S = Q · K^T  (BF16 vfmacc, stock)
    # 3) P = softmax(S)                   -> vfexp.v + vfredusum + vfdiv
    # 4) Quantize P to FP8                -> vfconv.bf16.fp8.v
    # 5) Load V block, dequant to BF16    -> vfconv.nvfp4.bf16.v
    # 6) O += P_fp8 · V_bf16              -> mixed-prec matmul (stock vfmacc + vfconv.fp8.bf16.v on V)
    ...

# Schedule (rough):
sched = flash_attn_tile_mixed
sched = divide_loop(sched, "n", BLK_N, ["no", "ni"], perfect=True)
sched = stage_mem(sched, "K_nvfp4[_]", "K_reg")
sched = set_memory(sched, "K_reg", SaturnRVV_M1)
sched = replace(sched, "for ni in _: K_reg[_] = K_nvfp4[_] #0", rvv_vle_nvfp4)
sched = replace(sched, "for i in _: K_bf16[_] = ... #0", vfconv_nvfp4_bf16_v)
# ... etc.
```

The four custom `@instr`s are sufficient to express the precision-routing FA kernel.
Stock RVV (vfmacc, vfredusum, vfdiv, vle/vse) covers the rest — these need the §3.4 VLEN-agnostic
declarations but are mechanical.

## 9. Open design questions (deferred)

1. **GCC RVV intrinsics naming.** Upstream GCC has `__riscv_vfmacc_vv_f32m1` but no `__riscv_vfconv_nvfp4_bf16`. Approach: `saturn_custom_asm.h` defines `SATURN_VFCONV_NVFP4_BF16(...)` macros that emit `asm volatile(".insn r ...")` with the funct6 we picked (`6'b111000`). Once GCC merges Saturn customs (likely never, since these are project-custom), the macros switch to intrinsics.
2. **NVFP4 packed-nibble layout.** Two NVFP4 nibbles per byte, low-nibble first. Need a stable convention; mirroring `gemmini-mx`'s nibble ordering simplifies future cross-comparison.
3. **Block-scale stride.** 1 E4M3 scale per 16 NVFP4 elements ⇒ scale buffer is 1/16 the data length. Exo can express this via separate buffer + `stride` assertions, but the loop nest will need `divide_loop(..., 16, perfect=True)` to keep the block alignment.
4. **vfexp output precision.** BF16-in, FP32-out is the current FU spec (LMUL-doubling). Alternative: BF16-in, BF16-out (saves output BW, risks softmax-sum overflow). Defer until Mo 5 numerical sweep.
5. **bf16 carrier vs first-class.** First-class `bf16` (§3.1) is the right paper claim. If the BF16 patch becomes a Y1 schedule risk, fall back to `uint16` carrier — the @instrs still type-check, but the paper's precision-routing-pass story weakens.

## 10. Validation plan for Mo 8 (compiler-viability checkpoint)

Mo 8 question: "Exo-generated mixed-prec FA within 10% of hand-coded?" Test plan:

1. **Mo 5 (Oct 2026):** Land §3 extensions in a fork of `exo-lang/exo`. Get a 10-line `@proc` to type-check + emit C using the §5 @instrs.
2. **Mo 6 (Nov 2026):** Manually schedule the FA tile (call `divide_loop`, `replace`, etc.) — produce one Exo-generated kernel + one hand-coded RVV intrinsics kernel. Run both on gem5 + Saturn-FU (FireSim if available).
3. **Mo 7 (Dec 2026):** Measure cycles-per-token on Llama-3.2-1B decode. Iterate scheduling to close any >10% gap.
4. **Mo 8 (Jan 2027):** Verdict. If gap > 10%, either (a) the §3 extensions are too coarse — refine memory class or extern semantics, (b) Exo's scheduling primitives miss something for FA tiling — propose a new primitive (paper-significant if it generalizes).

The §3 extensions are also the minimum viable upstream PR: BF16 type alone is ~30 LOC and broadly useful — credible for an Exo upstream PR independent of our Saturn customs.

## 11. Next-track impact

- F2/F3 (remaining vfconv lanes) → §5.2 and §5.3 above; the @instr decls hold whether the
  RTL latency is 1 or 2 cycles. Doing F2/F3 next still useful for Mo 4 but no longer compiler-gating.
- D follow-up: actually port the §4/§5 sketches into a real fork of `exo-lang/exo` (1–2 d). Want a
  type-checking smoke test before committing to Mo 5.
- H (FU-stub re-measurement) remains the highest-leverage Mo-2-strengthening track.

## 12. References

- Exo 2 paper: arXiv:2411.07211, ASPLOS 2025.
- Upstream Exo repo (`exo-lang/exo`, commit `2f5472d`, 2026-01-08):
  `src/exo/API.py:52` (`@instr` decorator), `src/exo/platforms/rvv.py` (175 LOC RVV stub),
  `src/exo/platforms/gemmini.py:284` (`@config` for vsetvl-like state),
  `src/exo/platforms/sve_vla.py:6` (VLA-style `@instr` with size param),
  `src/exo/frontend/typecheck.py:594-604` (primitive type table),
  `src/exo/libs/externs.py` (extern declaration pattern, `relu` / `select`).
- Saturn FU: `paper/fu_sketch.md` (the §Exo placeholder this artifact replaces).
- Precision config: `paper/precision_config.md`.

"""Q4_0 reference quantizer + the K64 levers L0..L4.

Everything here mirrors llama.cpp's `quantize_row_q4_0_ref` exactly, including
the detail that the *codes* are derived from the full-precision inverse scale
while the *stored* scale is the fp16-rounded value:

    d  = max / -8        (max = signed value of largest magnitude)
    id = 1/d             (full precision)
    store fp16(d)
    code = min(15, (int)(x*id + 8.5))      -> value = d*(code-8)

Shape convention used throughout
--------------------------------
    x     : (..., K)                 float32, K a multiple of `block`
    d     : (..., nblk)              float32, fp16-rounded (what a file stores)
    codes : (..., nblk, block)       uint8 in [0,15]

so `dequant_q4_0(d, codes)` rebuilds a (..., nblk, block) array whose reshape is
(..., K).
"""
from __future__ import annotations

import numpy as np

QK4_0 = 32


def fp16(x):
    """Round-trip through fp16, exactly as GGML_FP32_TO_FP16 does."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def _as_blocks(x: np.ndarray, block: int) -> np.ndarray:
    assert x.shape[-1] % block == 0, f"{x.shape} not divisible by block={block}"
    return x.reshape(*x.shape[:-1], x.shape[-1] // block, block)


def quantize_q4_0_ref(x: np.ndarray, block: int = QK4_0):
    """Reference Q4_0 quantizer over the last axis. Returns (d, codes)."""
    xr = _as_blocks(x, block)

    absx = np.abs(xr)
    amax_idx = np.argmax(absx, axis=-1)
    mx = np.take_along_axis(xr, amax_idx[..., None], axis=-1)[..., 0]   # (..., nblk)

    d = mx / -8.0
    d = np.where(d == 0, np.float32(0.0), d).astype(np.float32)
    id_ = np.where(d != 0, 1.0 / np.where(d == 0, 1.0, d), 0.0).astype(np.float32)

    d_stored = fp16(d)
    codes = _round_code(xr, id_)
    return d_stored, codes


def dequant_q4_0(d: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """value = d * (code - 8); returns (..., nblk, block)."""
    return (codes.astype(np.float32) - 8.0) * np.asarray(d, dtype=np.float32)[..., None]


def _round_code(xr: np.ndarray, id_: np.ndarray) -> np.ndarray:
    """code = min(15, (int)(x*id + 8.5)) with FMA contraction of `x*id + 8.5`.

    llama.cpp's reference quantizer is compiled with -ffp-contract=fast, so the
    expression `x[i]*id + 8.5f` becomes a single fused multiply-add: the product
    is kept exact and the sum is rounded once. That matters only for values that
    land exactly on a half-way boundary, but there it changes the code by +-1.

    Modelling it as float64 (where a float32 x float32 product is exact) plus a
    single rounding back to float32 reproduces the production file bit-exactly;
    plain mul-then-add in float32 does not (verified over all 132 Q4_0 tensors).
    """
    v = (xr.astype(np.float64) * id_.astype(np.float64)[..., None] + 8.5).astype(np.float32)
    return np.clip(np.floor(v), 0, 15).astype(np.uint8)


# ---------------------------------------------------------------------------
# K64 levers
# ---------------------------------------------------------------------------

def lever_L0_k32_to_k64(d32: np.ndarray, codes32: np.ndarray):
    """L0: re-encode an existing K32 Q4_0 tensor into K64 groups.

    d = max(d_a, d_b) (== amax/8 of the 64-wide group when the two halves are
    read as magnitudes) and code' = clamp(round((code-8)*d_i/d)+8, 0, 15).
    """
    rows = d32.shape[0]
    nb32 = d32.shape[1]
    assert nb32 % 2 == 0
    nb64 = nb32 // 2

    dd = d32.reshape(rows, nb64, 2)
    cc = codes32.reshape(rows, nb64, 2, 32)

    # pick the half whose |d| is larger, preserving its sign
    take_a = np.abs(dd[..., 0]) >= np.abs(dd[..., 1])
    d64 = np.where(take_a, dd[..., 0], dd[..., 1])

    ratio = dd / np.where(d64[..., None] == 0, 1.0, d64[..., None])
    scaled = (cc.astype(np.float32) - 8.0) * ratio[..., None]
    # round half away from zero (np.round is half-to-even and would bias this)
    new = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) + 8.0
    codes64 = np.clip(new, 0.0, 15.0).astype(np.uint8).reshape(rows, nb64, 64)
    return fp16(d64), codes64


def _assign_codes(xr: np.ndarray, d: np.ndarray) -> np.ndarray:
    """code = clamp(floor(x*id + 8.5), 0, 15) with the same FMA contraction."""
    dd = np.where(d == 0, np.float32(1.0), d).astype(np.float32)
    id_ = np.where(dd != 0, np.float32(1.0) / np.where(dd == 0, np.float32(1.0), dd), np.float32(0.0))
    return _round_code(xr, id_)


def _ls_refit_d(xr: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """LS-optimal d for fixed codes: d* = sum(x*c)/sum(c^2), c = code-8, re-rounded to fp16."""
    c = codes.astype(np.float64) - 8.0
    num = np.sum(xr.astype(np.float64) * c, axis=-1)
    den = np.sum(c * c, axis=-1)
    d = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)
    return fp16(d)


def _error_diffusion(xr: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Fixed-order (ascending index) first-order error feedback."""
    xd = xr.astype(np.float64)
    dd = np.where(d == 0, np.float32(1.0), d).astype(np.float32)
    id_ = np.where(dd != 0, np.float32(1.0) / np.where(dd == 0, np.float32(1.0), dd), np.float32(0.0))
    id64 = id_.astype(np.float64)
    out = np.empty(xr.shape, dtype=np.uint8)
    acc = np.zeros(xr.shape[:-1], dtype=np.float64)
    for i in range(xr.shape[-1]):
        v = xd[..., i] + acc
        c = np.clip(np.floor(v * id64 + 8.5), 0, 15)
        out[..., i] = c.astype(np.uint8)
        acc = v - (c - 8.0) * dd.astype(np.float64)
    return out


def lever_L1(x: np.ndarray, block: int = 64):
    return quantize_q4_0_ref(x, block=block)


def lever_L2(x: np.ndarray, iters: int = 3, block: int = 64):
    """L1 + iterated least-squares scale refit (scale re-rounded to fp16 each round)."""
    xr = _as_blocks(x, block)
    d, codes = quantize_q4_0_ref(x, block=block)
    for _ in range(iters):
        d = _ls_refit_d(xr, codes)
        codes = _assign_codes(xr, d)
    # end on a scale refit, so the returned d is LS-optimal for the returned codes
    d = _ls_refit_d(xr, codes)
    return d, codes


def lever_L3(x: np.ndarray, iters: int = 3, block: int = 64):
    """L2 + error-diffusion rounding."""
    xr = _as_blocks(x, block)
    d, codes = quantize_q4_0_ref(x, block=block)
    for _ in range(iters):
        d = _ls_refit_d(xr, codes)
        codes = _error_diffusion(xr, d)
    return d, codes


_L4_FACTORS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10)


def lever_L4(x: np.ndarray, iters: int = 3, block: int = 64, factors=_L4_FACTORS):
    """L3 + clip search: try several scales near amax/8, keep the lowest-SSE one."""
    xr = _as_blocks(x, block)
    d0, _ = quantize_q4_0_ref(x, block=block)

    best_sse = None
    best = None
    for f in factors:
        d = fp16(d0 * np.float32(f))
        codes = _error_diffusion(xr, d)
        for _ in range(iters):
            d = _ls_refit_d(xr, codes)
            codes = _error_diffusion(xr, d)
        rec = dequant_q4_0(d, codes)
        sse = np.sum((xr - rec) ** 2, axis=-1)          # (..., nblk)
        if best_sse is None:
            best_sse, best = sse, (d, codes)
        else:
            better = sse < best_sse
            best_sse = np.where(better, sse, best_sse)
            bd, bc = best
            best = (np.where(better, d, bd), np.where(better[..., None], codes, bc))
    return best


def lever_L4_noed(x: np.ndarray, iters: int = 3, block: int = 64, factors=_L4_FACTORS):
    """Clip search on top of L2 -- i.e. L4 with the (MSE-harmful) error diffusion removed.

    First-order error feedback shapes quantization noise rather than removing it:
    the error transfer function (1 - z^-1) has power gain 2, so total error power
    rises by ~3 dB. That is measured, not assumed (see PRECISION-REPORT.md).
    This variant is the one that actually minimises squared error.
    """
    xr = _as_blocks(x, block)
    d0, _ = quantize_q4_0_ref(x, block=block)

    best_sse = None
    best = None
    for f in factors:
        d = fp16(d0 * np.float32(f))
        codes = _assign_codes(xr, d)
        for _ in range(iters):
            d = _ls_refit_d(xr, codes)
            codes = _assign_codes(xr, d)
        d = _ls_refit_d(xr, codes)
        rec = dequant_q4_0(d, codes)
        sse = np.sum((xr - rec) ** 2, axis=-1)
        if best_sse is None:
            best_sse, best = sse, (d, codes)
        else:
            better = sse < best_sse
            best_sse = np.where(better, sse, best_sse)
            bd, bc = best
            best = (np.where(better, d, bd), np.where(better[..., None], codes, bc))
    return best

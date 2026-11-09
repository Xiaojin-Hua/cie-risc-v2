#!/usr/bin/env python3
"""Re-quantize the Q4_0_K64 tensors of a model at block width 64 or 128, leaving every
other tensor byte-identical.

Design note (why this is a clean controlled experiment)
-------------------------------------------------------
The base file is the *production* model, not the BF16 source. Only the 132 Q4_0
tensors are rewritten; all Q4_K / Q8_0 / F32 tensors are copied verbatim. Since
`k64_quant.quantize_q4_0_ref` reproduces the production Q4_0 blocks bit-exactly
from the BF16 source (verified: 132/132 tensors, scales AND codes), running with
`--block 32` yields a file whose quantized payload is identical to production.
The K32 arm is therefore the production model, and the K64 arm differs from it
*only* in the Q4_0 block width.

Usage
-----
  requantize_gguf.py --base PROD.gguf --src BF16.gguf --dst OUT.gguf \
                     --block 64 --lever L4noed
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_lite import GGUFReader, GGML_TYPE, decode_q4_0_blocks  # noqa: E402
import k64_quant as K  # noqa: E402

# ggml type ids written into the GGUF tensor directory
TYPE_Q4_0 = 2
TYPE_Q4_0_K64 = 43

LEVERS = {
    "L0": "re-encode the production K32 codes into K64 groups (lower bound)",
    "L1": "quantize from BF16 at block width 64, d = amax/8",
    "L2": "L1 + iterated least-squares scale refit",
    "L3": "L2 + error-diffusion rounding (task spec; +3 dB noise shaping)",
    "L4": "L3 + clip search (task spec)",
    "L4noed": "L2 + clip search, no error diffusion (best measured)",
    "ref": "reference quantizer at the requested width (no lever)",
}


def pack_q4_0(d: np.ndarray, codes: np.ndarray, block: int) -> bytes:
    """Pack (d, codes) into ggml block layout: fp16 scale then nibbles.

    Nibble convention generalises Q4_0: qs[j] low = elem j, high = elem j + block/2.
    """
    rows, nblk, blk = codes.shape
    assert blk == block
    half = block // 2
    c = codes.astype(np.uint8)
    if block == 64:
        # Store as two Q4_0-compatible 32-wide halves sharing one scale, so the
        # block is byte-identical to two block_q4_0 records with equal d. This
        # lets vec_dot delegate to the RVV-vectorised ggml_vec_dot_q4_0_q8_0.
        a, b = c[..., 0:32], c[..., 32:64]
        qs = np.concatenate([a[..., :16] | (a[..., 16:] << 4),
                             b[..., :16] | (b[..., 16:] << 4)], axis=-1)
    else:
        qs = c[..., :half] | (c[..., half:] << 4)         # (rows, nblk, half)

    out = np.empty((rows, nblk, 2 + half), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(rows, nblk, 2)
    out[:, :, 2:] = qs
    return out.tobytes()


def pack_q4_0_k128(d: np.ndarray, codes: np.ndarray) -> bytes:
    """Pack a 128-wide block as TWO K64 records that share one fp16 scale.

    The result is byte-for-byte a valid Q4_0_K64 payload of the same size as the
    64-wide layout ((n/64)*34 B), so the stock loader, repack and kernel run it
    unchanged -- only the numerics are K128 (one scale per 128 values).
    """
    rows, nblk, blk = codes.shape
    assert blk == 128, blk
    d_u8 = d.astype(np.float16).view(np.uint8).reshape(rows, nblk, 2)

    def rec(cc):                                   # cc: (rows, nblk, 64) -> 34 B
        a, b = cc[..., 0:32], cc[..., 32:64]
        qs = np.concatenate([a[..., :16] | (a[..., 16:] << 4),
                             b[..., :16] | (b[..., 16:] << 4)], axis=-1)
        out = np.empty((rows, nblk, 34), dtype=np.uint8)
        out[:, :, 0:2] = d_u8
        out[:, :, 2:] = qs
        return out

    return np.concatenate([rec(codes[..., 0:64]), rec(codes[..., 64:128])], axis=-1).tobytes()


def quantize_rows(lever: str, w: np.ndarray, block: int, d32=None, c32=None):
    if lever == "ref":
        return K.quantize_q4_0_ref(w, block=block)
    if lever == "L0":
        assert d32 is not None and block == 64
        return K.lever_L0_k32_to_k64(d32, c32)
    if lever == "L1":
        return K.lever_L1(w, block=block)
    if lever == "L2":
        return K.lever_L2(w, block=block)
    if lever == "L3":
        return K.lever_L3(w, block=block)
    if lever == "L4":
        return K.lever_L4(w, block=block)
    if lever == "L4noed":
        return K.lever_L4_noed(w, block=block)
    raise SystemExit(f"unknown lever {lever}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="model supplying metadata + all non-Q4_0 tensors")
    ap.add_argument("--src", required=True, help="BF16 source for the Q4_0 tensors")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--block", type=int, default=64, choices=[64, 128])
    ap.add_argument("--lever", default="ref", choices=sorted(LEVERS))
    ap.add_argument("--only", default="", help="comma-separated tensor name filter (debug)")
    args = ap.parse_args()

    base = GGUFReader(args.base)
    src = GGUFReader(args.src)

    q40 = [n for n in base.tensor_order
           if base.tensors[n].type_name in ("Q4_0", "Q4_0_K64")]
    if args.only:
        keep = set(args.only.split(","))
        q40 = [n for n in q40 if n in keep]
    new_type = TYPE_Q4_0_K64          # both widths are stored in the K64 layout

    if not q40:
        raise SystemExit("no Q4_0 tensors selected")

    # ---- plan: data blob for every tensor, in the base file's order --------
    blobs: dict[str, bytes] = {}
    t0 = time.time()
    for i, name in enumerate(base.tensor_order):
        t = base.tensors[name]
        if name in set(q40):
            w = src.read_float32(name)
            d32 = c32 = None
            if args.lever == "L0":
                d32, c32 = decode_q4_0_blocks(base.raw(name), t.n_elements // t.shape[0], t.shape[0])
            d, codes = quantize_rows(args.lever, w, args.block, d32, c32)
            d = np.asarray(d, dtype=np.float32)
            blobs[name] = (pack_q4_0(d, codes, args.block) if args.block == 64
                           else pack_q4_0_k128(d, codes))
            print(f"  [{i+1}/{len(base.tensor_order)}] quantized {name:46s} "
                  f"{t.shape} -> Q4_0_K{args.block} "
                  f"({len(blobs[name])} B)", flush=True)
        else:
            blobs[name] = base.raw(name)

    # ---- write ------------------------------------------------------------
    align = base.alignment
    with open(args.dst, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", base.version))
        f.write(struct.pack("<Q", base.n_tensors))
        f.write(struct.pack("<Q", base.n_kv))
        f.write(base.kv_bytes())

        # tensor directory: same names/shapes/order, possible new type, new offsets
        offset = 0
        offsets = {}
        for name in base.tensor_order:
            t = base.tensors[name]
            blob = blobs[name]
            if name in set(q40):
                blk = args.block
                exp = (t.n_elements // blk) * (2 + blk // 2) if blk == 64 else \
                      (t.n_elements // 128) * 68
                assert len(blob) == exp, (name, len(blob), exp)
                tid = new_type
            else:
                assert len(blob) == t.nbytes, (name, len(blob), t.nbytes)
                tid = t.type_id
            offsets[name] = offset
            f.write(struct.pack("<Q", len(name)))
            f.write(name.encode())
            f.write(struct.pack("<I", len(t.shape)))
            f.write(struct.pack(f"<{len(t.shape)}Q", *t.shape))
            f.write(struct.pack("<I", tid))
            f.write(struct.pack("<Q", offset))
            offset += len(blob)
            offset += -offset % align

        pad = -f.tell() % align
        f.write(b"\x00" * pad)
        ds = f.tell()                      # data section start (offsets are relative to this)
        for name in base.tensor_order:
            want = ds + offsets[name]
            here = f.tell()
            if here < want:
                f.write(b"\x00" * (want - here))
            assert f.tell() == want, (name, f.tell(), want)
            f.write(blobs[name])

    print(f"wrote {args.dst} ({os.path.getsize(args.dst)} B) in {time.time()-t0:.0f}s")
    base.close()
    src.close()


if __name__ == "__main__":
    main()

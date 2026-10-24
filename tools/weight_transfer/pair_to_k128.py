#!/usr/bin/env python3
"""Convert a Q4_0_K64 model whose 64-block PAIRS share a scale into a real Q4_0_K128 file.

The K128-equivalent arms produced by requantize_k128.py --block 128 store each 128-wide
block as two 34-byte K64 records with an identical fp16 scale. A real block_q4_0_k128 is
just `ggml_half d; uint8_t qs[64]`, i.e. the same scale followed by the two records' 32
nibble bytes, so the conversion is a lossless 68 B -> 66 B repack (no requantization).

Other tensors are copied verbatim; only the tensor type id and the offsets change.

Usage: pair_to_k128.py IN.gguf OUT.gguf
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_lite import GGUFReader  # noqa: E402

TYPE_Q4_0_K64 = 43
TYPE_Q4_0_K128 = 44


def convert_tensor(raw: bytes, n_rows: int, n_cols: int) -> bytes:
    """two 34-byte K64 records sharing d -> one 66-byte K128 block."""
    assert n_cols % 128 == 0, n_cols
    nblk = n_cols // 128
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, nblk, 68)
    out = np.empty((n_rows, nblk, 66), dtype=np.uint8)
    d0 = a[:, :, 0:2]
    d1 = a[:, :, 34:36]
    if not np.array_equal(d0, d1):
        sys.exit(f"pair scales differ at {np.count_nonzero(d0 != d1)} place(s); not a K128-equivalent file")
    out[:, :, 0:2] = d0
    out[:, :, 2:34] = a[:, :, 2:34]      # first  64 values
    out[:, :, 34:66] = a[:, :, 36:68]    # second 64 values
    return out.tobytes()


def main() -> None:
    src_path, dst_path = sys.argv[1], sys.argv[2]
    base = GGUFReader(src_path)

    blobs: dict[str, bytes] = {}
    for i, name in enumerate(base.tensor_order):
        t = base.tensors[name]
        raw = base.raw(name)
        if t.type_name == "Q4_0_K64":
            rows = t.n_elements // t.shape[0]
            blobs[name] = convert_tensor(raw, rows, t.shape[0])
        else:
            blobs[name] = raw
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(base.tensor_order)}", flush=True)

    align = base.alignment
    with open(dst_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", base.version))
        f.write(struct.pack("<Q", base.n_tensors))
        f.write(struct.pack("<Q", base.n_kv))
        f.write(base.kv_bytes())

        offset = 0
        offsets = {}
        for name in base.tensor_order:
            t = base.tensors[name]
            blob = blobs[name]
            if t.type_name == "Q4_0_K64":
                exp = (t.n_elements // 128) * 66
                assert len(blob) == exp, (name, len(blob), exp)
                tid = TYPE_Q4_0_K128
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
        ds = f.tell()
        for name in base.tensor_order:
            want = ds + offsets[name]
            if f.tell() < want:
                f.write(b"\x00" * (want - f.tell()))
            f.write(blobs[name])

    print(f"wrote {dst_path} ({os.path.getsize(dst_path)} B)")
    base.close()


if __name__ == "__main__":
    main()

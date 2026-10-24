"""Minimal, dependency-free GGUF reader (numpy only).

Written for the Q4_0 K64 precision study so that every number in the report
can be traced to raw bytes rather than to a third-party library version.

Only what the study needs: header/KV parsing, tensor directory, and raw tensor
byte access for F32 / F16 / BF16 / Q4_0 / Q4_K / Q8_0.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# ggml_type enum values (ggml/include/ggml.h)
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    43: "Q4_0_K64",  # added by this study (see patch_q4_0_k64.py)
}

# (block_size, type_size_in_bytes) for the types we touch
GGML_BLOCK = {
    "F32":  (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q8_0": (32, 34),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
    "Q4_0_K64": (64, 34),
}

_GGUF_VALUE_TYPE = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
    6: "f32", 7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64",
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple          # ggml order: shape[0] is the fastest-varying dim
    type_id: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE.get(self.type_id, f"?{self.type_id}")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        blk, tsize = GGML_BLOCK[self.type_name]
        return (self.n_elements // blk) * tsize


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.metadata: dict = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._parse()

    # -- parsing ---------------------------------------------------------
    def _parse(self) -> None:
        f = self.f
        magic, version = struct.unpack("<II", f.read(8))
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file (magic={magic:#x})")
        self.version = version
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        self.n_tensors = n_tensors
        self.n_kv = n_kv
        self.kv_start = f.tell()

        for _ in range(n_kv):
            key = self._read_string()
            vtype = struct.unpack("<I", f.read(4))[0]
            self.metadata[key] = self._read_value(vtype)
        self.kv_end = f.tell()

        infos = []
        for _ in range(n_tensors):
            name = self._read_string()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            tid = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            infos.append(TensorInfo(name, tuple(dims), tid, off))
        self.tensor_info_end = f.tell()

        align = self.metadata.get("general.alignment", 32)
        pos = f.tell()
        self.data_start = pos + (-pos % align)
        self.tensors = {t.name: t for t in infos}
        self.tensor_order = [t.name for t in infos]
        self.alignment = align

    def kv_bytes(self) -> bytes:
        """Raw bytes of the metadata KV section (re-emitted verbatim by the writer)."""
        self.f.seek(self.kv_start)
        return self.f.read(self.kv_end - self.kv_start)

    def _read_string(self) -> str:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, vtype: int):
        f = self.f
        name = _GGUF_VALUE_TYPE[vtype]
        if name == "str":
            return self._read_string()
        if name == "arr":
            etype = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            ename = _GGUF_VALUE_TYPE[etype]
            if ename == "str":
                return [self._read_string() for _ in range(count)]
            fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I",
                   "i32": "i", "f32": "f", "bool": "?", "u64": "Q", "i64": "q",
                   "f64": "d"}[ename]
            return list(struct.unpack(f"<{count}{fmt}", f.read(count * struct.calcsize(fmt))))
        fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I",
               "i32": "i", "f32": "f", "bool": "?", "u64": "Q", "i64": "q",
               "f64": "d"}[name]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    # -- data access -----------------------------------------------------
    def raw(self, name: str) -> bytes:
        t = self.tensors[name]
        self.f.seek(self.data_start + t.offset)
        return self.f.read(t.nbytes)

    def read_float32(self, name: str) -> np.ndarray:
        """Dequantize a tensor to float32 with the raw ggml row layout preserved.

        Returned array has shape (n_rows, n_cols) where n_rows = prod(shape[1:])
        and n_cols = shape[0]; this matches the on-disk block ordering.
        """
        t = self.tensors[name]
        buf = self.raw(name)
        rows = t.n_elements // t.shape[0]
        return decode_blocks(t.type_name, buf, rows, t.shape[0])

    def close(self) -> None:
        self.f.close()


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode_blocks(type_name: str, buf: bytes, n_rows: int, n_cols: int) -> np.ndarray:
    """Decode raw bytes into a (n_rows, n_cols) float32 array."""
    if type_name in ("F32", "F16", "BF16"):
        dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[type_name]
        a = np.frombuffer(buf, dtype=dt).reshape(n_rows, n_cols)
        if type_name == "BF16":
            return _bf16_to_f32(a)
        return a.astype(np.float32)

    if type_name == "Q4_0":
        blk, tsize = 32, 18
        nblk = n_cols // blk
        a = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, nblk, tsize)
        d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
        qs = a[:, :, 2:]
        lo = (qs & 0x0F).astype(np.int32) - 8
        hi = (qs >> 4).astype(np.int32) - 8
        codes = np.concatenate([lo, hi], axis=2)          # (rows, nblk, 32)
        return (codes * d[:, :, None]).reshape(n_rows, n_cols)

    if type_name == "Q4_0_K64":
        # 34 B per 64 values: fp16 d + qs[32]; qs[0:16] = values 0..31, qs[16:32] = 32..63,
        # each byte low nibble = first half, high nibble = second half (same as Q4_0 twice)
        blk, tsize = 64, 34
        nblk = n_cols // blk
        a = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, nblk, tsize)
        d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
        qs = a[:, :, 2:]                              # 32 B = two 32-wide halves
        lo = (qs & 0x0F).astype(np.int32) - 8
        hi = (qs >> 4).astype(np.int32) - 8
        # half 0 = values 0..31 (low nibbles 0..15, high nibbles 16..31),
        # half 1 = values 32..63 -- see dequantize_row_q4_0_k64 in ggml-quants.c
        codes = np.concatenate([lo[:, :, 0:16], hi[:, :, 0:16],
                                lo[:, :, 16:32], hi[:, :, 16:32]], axis=2)
        return (codes * d[:, :, None]).reshape(n_rows, n_cols)

    if type_name == "Q8_0":
        # ggml block_q8_0: fp16 d + 32 int8 quants, value = d * q
        blk, tsize = 32, 34
        nblk = n_cols // blk
        a = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, nblk, tsize)
        d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
        q = a[:, :, 2:].view(np.int8).astype(np.float32)
        return (q * d[:, :, None]).reshape(n_rows, n_cols)

    raise NotImplementedError(f"decode not implemented for {type_name}")


def decode_q4_0_blocks(buf: bytes, n_rows: int, n_cols: int):
    """Return (d, codes) for a Q4_0 tensor.

    d     : (n_rows, n_cols//32) float32 (already fp16-rounded, as stored)
    codes : (n_rows, n_cols//32, 32) int8 in [0, 15]  (raw nibbles, unsigned)
    """
    blk, tsize = 32, 18
    nblk = n_cols // blk
    a = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, nblk, tsize)
    d = a[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = a[:, :, 2:]
    lo = (qs & 0x0F).astype(np.uint8)
    hi = (qs >> 4).astype(np.uint8)
    codes = np.concatenate([lo, hi], axis=2)
    return d, codes

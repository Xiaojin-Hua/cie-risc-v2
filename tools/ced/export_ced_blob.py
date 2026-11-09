#!/usr/bin/env python3
"""Write a CED projection blob (CEDPROJ1) from trained K/V matrices.

The board loader (src/models/llama-ced.h) expects, all little-endian:

    offset  0 : char   magic[8]  = "CEDPROJ1"
    offset  8 : uint32 n_layers_calibrated
    offset 12 : uint32 hidden_size
    offset 16 : uint32 kv_dim
    offset 20 : uint32 split_layer
    offset 24 : uint32 tail
    offset 28 : uint32 layer_ids[n_layers_calibrated]
    then, per calibrated layer in layer_ids order:
        fp16 K[kv_dim, hidden_size]   row-major, ne[0] = hidden_size
        fp16 V[kv_dim, hidden_size]

`check_export.py` verifies the result reproduces the GPU-side projected K/V.

Usage:
    python3 export_ced_blob.py --src proj.pt --out ced_proj.bin \
        --hidden 2048 --kv 512 --split 14 --tail 512
    # --src may be a .pt (torch), .npz (numpy) or .npz-style dict holding
    #   {"k": {layer: array[kv,hidden]}, "v": {...}}   or   {(layer,"k"): array}
"""
import argparse, struct, sys
import numpy as np

MAGIC = b"CEDPROJ1"

def load_src(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import torch
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    return {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in obj.items()}

def collect(d, kv_dim, hidden):
    """Normalise whatever the trainer saved into {layer: {k: arr, v: arr}}."""
    out = {}
    def put(layer, kind, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (kv_dim, hidden):
            if arr.T.shape == (kv_dim, hidden):
                arr = arr.T                      # accept [hidden, kv_dim] too
            else:
                raise SystemExit(f"layer {layer} {kind}: shape {arr.shape} != {(kv_dim, hidden)}")
        out.setdefault(int(layer), {})[kind] = arr
    # form 1: "k" -> {layer: arr}
    for kind in ("k", "v"):
        if kind in d and isinstance(d[kind], dict):
            for layer, arr in d[kind].items():
                put(layer, kind, arr)
    # form 2: "blk.14.k.weight" / "14.k" style keys
    for name, arr in d.items():
        if name in ("k", "v"):
            continue
        parts = str(name).replace(".weight", "").split(".")
        nums = [p for p in parts if p.isdigit()]
        # the trainer names them k_norm / v_proj (K and V projections at the
        # full-attention layers); accept plain k / v as well
        kinds = ["k" if p.startswith("k") else "v" for p in parts if p.startswith(("k", "v"))]
        if nums and kinds:
            put(nums[-1], kinds[-1], arr)
    if not out:
        raise SystemExit("could not find K/V matrices in the source file")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--tail", type=int, required=True)
    a = ap.parse_args()

    per_layer = collect(load_src(a.src), a.kv, a.hidden)
    ids = sorted(per_layer)
    missing = [l for l in ids if "k" not in per_layer[l] or "v" not in per_layer[l]]
    if missing:
        raise SystemExit(f"layers without both K and V: {missing}")

    with open(a.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIII", len(ids), a.hidden, a.kv, a.split, a.tail))
        f.write(struct.pack(f"<{len(ids)}I", *ids))
        for l in ids:
            f.write(per_layer[l]["k"].astype(np.float16).tobytes())
            f.write(per_layer[l]["v"].astype(np.float16).tobytes())
    print(f"wrote {a.out}: {len(ids)} layer(s) {ids}, "
          f"hidden={a.hidden} kv={a.kv} split={a.split} tail={a.tail}")
    print("verify with: python3 check_export.py --blob", a.out)

if __name__ == "__main__":
    main()

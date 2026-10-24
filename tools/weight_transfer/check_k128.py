#!/usr/bin/env python3
"""Verify a board GGUF is the expected Q4_0_K128 layout.

Reference (Qwen3.5-2B ssm_all_k128.gguf): 186 tensors of type 44 (Q4_0_K128),
1 tensor of type Q4_K (output.weight), the rest F32.

Usage: python3 check_k128.py model.gguf [--expect-k128 N]
Exit code 0 when the layout matches, 1 otherwise.
"""
import argparse, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_lite import GGUFReader

K128, Q4K, F32 = 44, 12, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--expect-k128", type=int, default=0,
                    help="expected number of Q4_0_K128 tensors (0 = just require > 0)")
    a = ap.parse_args()
    r = GGUFReader(a.gguf)
    c = collections.Counter()
    for n in r.tensor_order:
        c[getattr(r.tensors[n], "type_id", -1)] += 1
    n128, nq4k, nf32 = c.get(K128, 0), c.get(Q4K, 0), c.get(F32, 0)
    other = {k: v for k, v in c.items() if k not in (K128, Q4K, F32)}
    ok = True
    print(f"{a.gguf}")
    print(f"  Q4_0_K128 (44) : {n128}")
    print(f"  Q4_K      (12) : {nq4k}   (expected 1: output.weight)")
    print(f"  F32        (0) : {nf32}")
    if other:
        print(f"  OTHER          : {other}   <- not expected")
        ok = False
    if n128 == 0:
        print("  FAIL: no Q4_0_K128 tensors — the pair_to_k128.py step was skipped?")
        ok = False
    if a.expect_k128 and n128 != a.expect_k128:
        print(f"  FAIL: expected {a.expect_k128} K128 tensors, found {n128}")
        ok = False
    if nq4k != 1:
        print(f"  FAIL: expected exactly 1 Q4_K tensor (output.weight), found {nq4k}")
        ok = False
    print("  OK" if ok else "  LAYOUT MISMATCH")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

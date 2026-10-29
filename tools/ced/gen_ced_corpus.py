#!/usr/bin/env python3
"""Build the token-shard corpus that ced_final_fixed.py consumes.

The trainer reads an index JSON:
    {"shards": [{"path": "/abs/shard_000.npy"}, ...]}
    tr = concat(np.load(all but last));  ev = np.load(last)

so each shard is just a 1-D array of token ids, and the LAST shard is the held-out
evaluation set.

Usage:
    python3 gen_ced_corpus.py --model <HF model dir> --text a.txt [b.txt ...] \
        --out DIR [--shard-tokens 2000000] [--eval-shards 1]
"""
import argparse, glob, json, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (for the tokenizer)")
    ap.add_argument("--text", nargs="+", required=True, help="plain-text files or globs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--shard-tokens", type=int, default=2_000_000)
    ap.add_argument("--eval-shards", type=int, default=1,
                    help="how many trailing shards are held out (default 1)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    files = []
    for pat in a.text:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not files:
        raise SystemExit("no input text found")

    os.makedirs(a.out, exist_ok=True)
    ids = []
    for p in files:
        txt = open(p, encoding="utf-8", errors="ignore").read()
        ids.extend(tok(txt)["input_ids"])
        print(f"  {p}: {len(ids)} tokens so far", flush=True)

    arr = np.asarray(ids, dtype=np.int32)
    n = len(arr)
    cuts = list(range(0, n, a.shard_tokens))
    shards = []
    for i, s in enumerate(cuts):
        chunk = arr[s:s + a.shard_tokens]
        if len(chunk) < a.shard_tokens // 4 and i > 0:
            break                       # drop a tiny tail shard
        path = os.path.join(a.out, f"shard_{i:03d}.npy")
        np.save(path, chunk)
        shards.append({"path": path, "tokens": int(len(chunk))})
        print(f"  wrote {path} ({len(chunk)} tokens)", flush=True)

    if len(shards) <= a.eval_shards:
        raise SystemExit("not enough shards; lower --shard-tokens")
    index = {"shards": shards, "total_tokens": int(sum(s["tokens"] for s in shards)),
             "eval_shards": a.eval_shards}
    ip = os.path.join(a.out, "index.json")
    json.dump(index, open(ip, "w"), indent=1)
    print(f"wrote {ip}: {len(shards)} shards, {index['total_tokens']} tokens, "
          f"eval = last {a.eval_shards}")

if __name__ == "__main__":
    main()

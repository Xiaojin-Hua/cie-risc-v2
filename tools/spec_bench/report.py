#!/usr/bin/env python3
"""Aggregate spec_bench measurements into a comparison table.

    python3 report.py results/*.json
"""
import glob, json, sys
from collections import defaultdict

rows = []
for pat in (sys.argv[1:] or ["results/*.json"]):
    for f in sorted(glob.glob(pat)):
        try:
            rows.append(json.load(open(f)))
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)

if not rows:
    raise SystemExit("no measurements found")

# group by (model, prompts, max_tokens); baseline = spec == "none"
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["prompts"], r["max_tokens"])].append(r)

print(f"{'model':22s} {'prompts':8s} {'tok':>4s} {'config':26s} {'t/s':>7s} {'gain':>7s} "
      f"{'acc':>6s} {'len':>5s} {'ok':>3s}")
print("-" * 100)
for key, rs in groups.items():
    base = next((r for r in rs if r["spec"] == "none" and r["out_tps"]), None)
    base_tps = base["out_tps"] if base else None
    for r in sorted(rs, key=lambda x: (x["spec"] != "none", x["tag"])):
        tps = r["out_tps"]
        gain = f"{(tps/base_tps-1)*100:+.1f}%" if (tps and base_tps) else "-"
        spec = "baseline" if r["spec"] == "none" else r["spec"][:24]
        ok = "BAD" if r.get("degenerate") else "ok"
        print(f"{r['model'][:22]:22s} {r['prompts']:8s} {r['max_tokens']:4d} {spec:26s} "
              f"{(tps if tps else 0):7.2f} {gain:>7s} "
              f"{(r['acceptance'] if r['acceptance'] is not None else 0):6.3f} "
              f"{(r['mean_len'] if r['mean_len'] is not None else 0):5.2f} {ok:>3s}")
print()
print("'BAD' = output looked degenerate (unique 5-gram ratio < 0.5): discard that row.")

#!/usr/bin/env python3
"""Aggregate the fork's [spxnode] per-op trace lines into per-op time shares.

Usage: python3 op_share.py <log> [<log> ...]
Prints, per log: graph count, op-table total, per-op share, and the MUL_MAT
breakdown by shape.
"""
import re
import sys
from collections import defaultdict

NODE = re.compile(
    r"\[spxnode\]\s+stage=\d+\s+idx=\s*(\d+)\s+op=(\S+)\s+nth=(\d+)\s+us=\s*(\d+)\s+ne=\[([^\]]*)\]"
)
GRAPH = re.compile(r"\[spxgraph\]")


def parse(path):
    """Return [(section_title, [graph, ...]), ...]; sections split on '#####' markers."""
    sections = []
    title = "(start)"
    graphs = []
    cur = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("#####"):
            if cur:
                graphs.append(cur)
                cur = []
            if graphs:
                sections.append((title, graphs))
                graphs = []
            t = line.strip().lstrip("#").strip(" -")
            if "SNAP" in t or "arm" in t or "n_prompt" in t or "====" in t:
                title = t
            continue
        if GRAPH.search(line):
            if cur:
                graphs.append(cur)
            cur = []
            continue
        m = NODE.search(line)
        if m:
            idx, op, nth, us, ne = m.groups()
            cur.append((op, int(us), ne.strip()))
    if cur:
        graphs.append(cur)
    if graphs:
        sections.append((title, graphs))
    return sections


def report(path):
    sections = parse(path)
    if not sections:
        print(f"{path}: no [spxnode] lines")
        return
    for title, gs in sections:
        agg(title, gs)


def agg(title, graphs):
    full = [g for g in graphs if len(g) > 50]
    graphs = full or graphs
    n_graphs = len(graphs)

    # average a graph's op sums across graphs (graphs may differ in size)
    op_us = defaultdict(float)
    shape_us = defaultdict(float)
    mulmat_us = 0.0
    graph_total = 0.0
    for g in graphs:
        tot = sum(us for _, us, _ in g)
        graph_total += tot
        for op, us, ne in g:
            op_us[op] += us
            if op == "MUL_MAT":
                mulmat_us += us
                shape_us[ne] += us
    scale = 1.0 / n_graphs
    total = graph_total * scale
    print(f"\n### {title}")
    print(f"    graphs={n_graphs}  nodes/graph~{sum(len(g) for g in graphs)/n_graphs:.0f}"
          f"  op-table total={total/1e6:.3f} s/graph")
    print(f"    {'op':20s} {'ms/graph':>10s} {'share':>8s}")
    for op, us in sorted(op_us.items(), key=lambda kv: -kv[1]):
        ms = us * scale / 1000.0
        print(f"    {op:20s} {ms:10.2f} {100.0*us/graph_total:7.2f}%")
    print(f"    {'-- MUL_MAT by shape':20s}")
    for ne, us in sorted(shape_us.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {'[' + ne + ']':20s} {us*scale/1000.0:10.2f} "
              f"{100.0*us/mulmat_us if mulmat_us else 0:7.2f}% of MUL_MAT"
              f"  {100.0*us/graph_total:6.2f}% of all")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)

#!/usr/bin/env python3
"""Generate the spec_bench prompt sets.

All prompts are INSTRUCT form on purpose: raw text continuation makes the model
degenerate into a repetition loop, and ngram then scores 90%+ acceptance on the
loop — a measurement artefact, not a speedup (see README section 1).

    python3 gen_prompts.py --which qa|json|summary --out prompts/qa.txt [--corpus wiki.raw]
"""
import argparse, os, re

QA = [
    "用一句话解释什么是 RISC-V 向量扩展。",
    "请说明 CPU 的 L2 缓存和内存带宽之间是什么关系。",
    "解释一下什么是量化，为什么 4bit 量化能加速推理。",
    "请用两三句话说明投机解码（speculative decoding）的原理。",
    "什么是 KV cache？为什么长上下文会变慢？",
    "请解释 prefill 和 decode 两个阶段的区别。",
    "为什么大模型的 decode 阶段通常是内存带宽受限的？",
    "请说明混合专家（MoE）模型相比稠密模型在推理上的优势。",
]

NOTHINK = "直接输出 JSON，不要思考过程，不要解释。"

JSON = [
    "直接输出 JSON，不要思考过程，不要解释。请输出一个 JSON 数组，包含 15 个对象，每个对象描述一家芯片公司，字段为 name、founded、hq。",
    "请输出一个 JSON 数组，包含 15 个对象，每个对象描述一款 RISC-V 开发板，字段为 board、soc、ram_gb。只输出 JSON。",
    "请输出一个 JSON 对象，包含 15 个键值对，键是中文成语，值是其英文翻译。只输出 JSON。",
    "请输出一个 JSON 数组，包含 15 个对象，每个对象是一门编程语言，字段为 lang、year、paradigm。只输出 JSON。",
    "请输出一个 JSON 数组，包含 15 个对象，每个对象是一个城市，字段为 city、country、population_m。只输出 JSON。",
    "请输出一个 JSON 对象，包含 15 个字段，字段名是 Linux 命令，值是它的一句话用途。只输出 JSON。",
]

SUMMARY_PREFIX = "请用中文概括下面这段文字的主要内容，不要超过 100 字：\n\n"

def clean_paragraphs(path, n):
    """Pick instructional-looking prose: no wiki markup, no template noise."""
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if len(s) < 220 or "@-" in s or "= =" in s or s.startswith("="):
            continue
        if re.search(r"[{}<>]{2,}", s):
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["qa", "json", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/home/bianbu/k64-precision/corpus/wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--count", type=int, default=6, help="how many summary prompts")
    a = ap.parse_args()

    if a.which == "qa":
        lines = QA
    elif a.which == "json":
        lines = JSON
    else:
        paras = clean_paragraphs(a.corpus, a.count)
        if not paras:
            raise SystemExit(f"no usable paragraphs in {a.corpus}")
        lines = [SUMMARY_PREFIX + p[:900] for p in paras]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(lines):
            f.write(p.replace("\n", " ").strip() + "\n")
    print(f"wrote {a.out}: {len(lines)} prompts ({a.which})")

if __name__ == "__main__":
    main()

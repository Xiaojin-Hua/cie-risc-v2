#!/usr/bin/env bash
# run_bench.sh — one speculative-decoding measurement via evalscope perf.
#
#   ./run_bench.sh --model M.gguf --prompts qa --tag base_qa [--spec ""] [--max-tokens 128]
#
# Writes <outdir>/<tag>.json with throughput, TTFT/TPOT, the server's draft acceptance
# and a degeneracy flag. See README.md for why each of those is required.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# board-side defaults; override with env
B="${LLAMA_BIN:-/home/bianbu/cie-risc-v2/llama.cpp/build-k1/bin}"
EVALSCOPE="${EVALSCOPE:-/home/bianbu/myenv/bin/evalscope}"
TOKDIR="${TOKDIR:-/home/bianbu/kwork/specperf/tok}"
OUT="${OUTDIR:-$HERE/results}"
PORT="${PORT:-8080}"
CORPUS="${CORPUS:-/home/bianbu/k64-precision/corpus/wikitext-2-raw/wiki.test.raw}"

MODEL=""; SPEC=""; PROMPTS="qa"; MAXT=128; TAG=""; THREADS=4; NUMBER=3

while [ $# -gt 0 ]; do case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --spec) SPEC="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --max-tokens) MAXT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --number) NUMBER="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
esac; done
[ -n "$MODEL" ] || { echo "error: --model is required" >&2; exit 2; }
[ -n "$TAG" ]   || { echo "error: --tag is required" >&2; exit 2; }
mkdir -p "$OUT" "$HERE/logs"

# prompt file: generate the summary set on demand from the wikitext corpus
PF="$HERE/prompts/$PROMPTS.txt"
if [ ! -f "$PF" ]; then
    echo "[spec_bench] prompt set '$PROMPTS' not found, generating from $CORPUS"
    python3 "$HERE/gen_prompts.py" --which "$PROMPTS" --corpus "$CORPUS" --out "$PF" || exit 1
fi

pkill -9 -x llama-server 2>/dev/null; sleep 3
for i in $(seq 1 20); do pgrep -x llama-server >/dev/null || break; sleep 1; done
[ "$(pgrep -x llama-server | wc -l)" -eq 0 ] || { echo "FATAL stale server"; exit 1; }

SRVLOG="$HERE/logs/srv_$TAG.log"
env -u LLAMA_QWEN35_CED LLAMA_QWEN35_CED="${LLAMA_QWEN35_CED:-}" \
    LD_LIBRARY_PATH="$B" "$B/llama-server" -m "$MODEL" -t "$THREADS" -c 4096 \
    --chat-template-kwargs '{"enable_thinking":false}' -b 512 -ub 512 -fa 1 --host 127.0.0.1 --port "$PORT" $SPEC > "$SRVLOG" 2>&1 &
for i in $(seq 1 240); do
    case "$(curl -s "http://127.0.0.1:$PORT/health")" in *ok*) break;; esac; sleep 2
done
case "$(curl -s "http://127.0.0.1:$PORT/health")" in *ok*) :;; *)
    echo "FATAL server health"; tail -5 "$SRVLOG"; pkill -9 -x llama-server; exit 1;; esac

rm -rf "$HERE/outputs"; mkdir -p "$HERE/outputs"
"$EVALSCOPE" perf --model spec_bench --api openai \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --dataset line_by_line --dataset-path "$PF" \
    --tokenizer-path "$TOKDIR" --parallel 1 --number "$NUMBER" \
    --max-tokens "$MAXT" --min-tokens "$MAXT" --max-prompt-length 2048 \
    --stream --outputs-dir "$HERE/outputs" > "$HERE/logs/eval_$TAG.log" 2>&1
RC=$?

ACC=$(grep -a "draft acceptance" "$SRVLOG" | tail -1 | sed -n 's/.*acceptance = \([0-9.]*\).*mean len = *\([0-9.]*\).*/\1 \2/p')
OUTTPS=$(grep -ahoE "Output Throughput[^0-9]*[0-9.]+" "$HERE/logs/eval_$TAG.log" | tail -1 | grep -oE "[0-9.]+$")
TTFT=$(grep -ahoE "TTFT[^0-9]*[0-9.]+" "$HERE/logs/eval_$TAG.log" | tail -1 | grep -oE "[0-9.]+$")
TPOT=$(grep -ahoE "TPOT[^0-9]*[0-9.]+" "$HERE/logs/eval_$TAG.log" | tail -1 | grep -oE "[0-9.]+$")

python3 - "$TAG" "$MODEL" "$SPEC" "$PROMPTS" "$MAXT" "${OUTTPS:-}" "${TTFT:-}" "${TPOT:-}" "${ACC:-}" "$RC" "$HERE/outputs" "$OUT/$TAG.json" <<'PY'
import json, sys, glob, os
tag, model, spec, prompts, maxt, tps, ttft, tpot, acc, rc, outdir, dest = sys.argv[1:13]
acc_v, len_v = (acc.split() + ["", ""])[:2] if acc else ("", "")
# degeneracy check: unique 5-gram ratio of the generated text must stay reasonable
degenerate = None
txts = glob.glob(os.path.join(outdir, "**", "*.txt"), recursive=True)
if txts:
    words = open(txts[0], encoding="utf-8", errors="ignore").read().split()
    if len(words) >= 8:
        grams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        degenerate = (len(set(grams)) / max(1, len(grams))) < 0.5
rec = dict(tag=tag, model=os.path.basename(model), spec=spec or "none", prompts=prompts,
           max_tokens=int(maxt),
           out_tps=float(tps) if tps else None,
           ttft_ms=float(ttft) if ttft else None,
           tpot_ms=float(tpot) if tpot else None,
           acceptance=float(acc_v) if acc_v else None,
           mean_len=float(len_v) if len_v else None,
           degenerate=degenerate, evalscope_rc=int(rc))
json.dump(rec, open(dest, "w"), indent=1)
print(json.dumps(rec, indent=1))
if degenerate:
    print("WARNING: output looks degenerate (unique 5-gram ratio < 0.5) — the measurement "
          "is INVALID, see README section 1", file=sys.stderr)
PY

pkill -9 -x llama-server 2>/dev/null; sleep 2

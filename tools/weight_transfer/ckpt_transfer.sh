#!/usr/bin/env bash
# ckpt_transfer.sh <source_path> <output_path>
#
# Convert raw checkpoint weights into the board's Q4_0_K128 GGUF — byte-layout identical
# to /home/bianbu/kwork/nongemm/ssm_all_k128.gguf (2B) and Qwen3.5-4B-k128-v2.gguf (4B).
#
#   <source_path>  an HF model directory, or a high-precision GGUF (f16 / bf16 / q8_0)
#   <output_path>  the .gguf to write
#
# Pipeline (four steps; step 3 and step 4 are both required):
#   1. HF dir -> f16 GGUF                       convert_hf_to_gguf.py
#   2. f16    -> q4_0 base (K64 tensors)        llama-quantize, output.weight forced to Q4_K
#   3. base + high-precision -> K64 records     requantize_k128.py --block 128
#      whose 64-block PAIRS share one fp16 scale. NOTE: this file is still TYPED K64(43);
#      only its numerics are K128. Skipping step 4 is the classic mistake: it runs the
#      K64 kernels and loses ~18% prefill.
#   4. K64(paired) -> real Q4_0_K128 (44)       pair_to_k128.py  (lossless 68B -> 66B)
#
# Required environment:
#   LLAMA_ROOT      llama.cpp checkout with convert_hf_to_gguf.py + a built llama-quantize
#                   (override the binary with LLAMA_QUANTIZE=...)
#   PY              python3 with numpy (default python3)
#
# Helpers that must sit next to this script (copy them together to another machine):
#   requantize_k128.py  k64_quant.py  gguf_lite.py  pair_to_k128.py  check_k128.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
LLAMA_ROOT="${LLAMA_ROOT:-}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-}"

die()  { echo "error: $*" >&2; exit 2; }
note() { echo "[ckpt_transfer] $*"; }

[ $# -eq 2 ] || { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
SRC="$1"; DST="$2"
[ -e "$SRC" ] || die "source not found: $SRC"
for f in requantize_k128.py k64_quant.py gguf_lite.py pair_to_k128.py check_k128.py; do
    [ -f "$HERE/$f" ] || die "missing helper: $HERE/$f"
done
"$PY" -c 'import numpy' 2>/dev/null || die "python3 with numpy is required"

# temp files must never survive a failure (a 1.2 GB base got leaked once)
TMP=()
cleanup() { for f in ${TMP[@]+"${TMP[@]}"}; do rm -f "$f"; done; }
trap cleanup EXIT

# ---- 1. get a high-precision GGUF
if [ -d "$SRC" ]; then
    [ -n "$LLAMA_ROOT" ] || die "LLAMA_ROOT is required for an HF directory"
    [ -f "$LLAMA_ROOT/convert_hf_to_gguf.py" ] || die "$LLAMA_ROOT/convert_hf_to_gguf.py not found"
    SRC_HI="$(mktemp -u /tmp/ckpt_f16_XXXXXX.gguf)"; TMP+=("$SRC_HI")
    note "1/4 HF -> f16 GGUF : $SRC -> $SRC_HI"
    "$PY" "$LLAMA_ROOT/convert_hf_to_gguf.py" "$SRC" --outfile "$SRC_HI" --outtype f16 \
        || die "convert_hf_to_gguf.py failed"
else
    case "$SRC" in *.gguf) SRC_HI="$SRC" ;; *) die "source must be an HF dir or a .gguf" ;; esac
    note "1/4 source is already GGUF: $SRC_HI"
    note "    must still be high precision (f16/bf16/q8_0) — a q4 file cannot be requantized"
fi

# ---- 2. q4_0 base, output.weight = Q4_K (this is what the reference models use)
if [ -z "$LLAMA_QUANTIZE" ]; then
    [ -n "$LLAMA_ROOT" ] || die "LLAMA_ROOT or LLAMA_QUANTIZE is required"
    for c in "$LLAMA_ROOT/build/bin/llama-quantize" "$LLAMA_ROOT/build-k1/bin/llama-quantize" \
             "$LLAMA_ROOT/build-x86/bin/llama-quantize"; do
        [ -x "$c" ] && LLAMA_QUANTIZE="$c" && break
    done
fi
[ -n "$LLAMA_QUANTIZE" ] && [ -x "$LLAMA_QUANTIZE" ] || die "llama-quantize not found (set LLAMA_QUANTIZE=...)"
BASE="$(mktemp -u /tmp/ckpt_base_XXXXXX.gguf)"; TMP+=("$BASE")
export LD_LIBRARY_PATH="$(cd "$(dirname "$LLAMA_QUANTIZE")" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
note "2/4 q4_0 base (output.weight -> Q4_K) : $BASE"
if ! "$LLAMA_QUANTIZE" --allow-requantize --output-tensor-type q4_k "$SRC_HI" "$BASE" q4_0 \
        >/tmp/ckpt_quantize.log 2>&1; then
    tail -5 /tmp/ckpt_quantize.log >&2
    die "llama-quantize failed (see /tmp/ckpt_quantize.log)"
fi

# ---- 3. re-quantize the K64 tensors with 128-wide shared scales
STEP3="$(mktemp -u /tmp/ckpt_k128_XXXXXX.gguf)"; TMP+=("$STEP3")
note "3/4 requantize --block 128 (numerics K128, still typed K64) : $STEP3"
"$PY" "$HERE/requantize_k128.py" --base "$BASE" --src "$SRC_HI" --dst "$STEP3" --block 128 \
    || die "requantize_k128.py failed"

# ---- 4. lossless repack to the real Q4_0_K128 type
note "4/4 pair_to_k128 -> real Q4_0_K128 : $DST"
"$PY" "$HERE/pair_to_k128.py" "$STEP3" "$DST" || die "pair_to_k128.py failed"

# ---- verify the layout matches the reference models
note "verifying layout"
"$PY" "$HERE/check_k128.py" "$DST" || die "layout check failed (see above)"

note "done: $(ls -la "$DST" | awk '{print $5}') bytes -> $DST"
note "reference sizes: 2B ssm_all_k128.gguf = 1007028256 B, 4B k128-v2 = 2212707232 B"

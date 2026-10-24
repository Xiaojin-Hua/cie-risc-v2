#!/usr/bin/env bash
# ced_train.sh — train a CED K/V-projection blob for Qwen3.5-2B or 4B.
#
#   ./ced_train.sh --size 2b --out DIR --tag name [extra args passed to the trainer]
#   ./ced_train.sh --size 4b --out DIR --tag name
#
# The two parameter sets below are the only size-specific knowledge; everything else is
# forwarded to ced_final_fixed.py (--split-layer --tail --ctx --windows --train-steps
# --batch --lr --capacity ...).
#
# Sizes (from the model configs):
#   2B : hidden 2048, 24 layers, kv_dim 2*256=512  , full-attn at 3,7,11,15,19,23
#   4B : hidden 2560, 32 layers, kv_dim 4*256=1024 , full-attn at 3,7,11,...,31
# The split_layer starts the decoder half. Keep it on a GDN layer: 2B=14, 4B=17
# (4B's 15 and 19 are full-attention layers; a split there changes which CED path
#  the decoder begins with, so sweep 16/17/18 rather than 19).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

die()  { echo "error: $*" >&2; exit 2; }
need() { [ -n "${2:-}" ] || die "$1 is required"; }

SIZE=""; OUT=""; TAG=""; CORPUS=""; MODEL=""
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)   SIZE="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        --tag)    TAG="$2"; shift 2 ;;
        --corpus-index) CORPUS="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

need --size "$SIZE"; need --out "$OUT"
case "$SIZE" in
    2b|2B)
        DEF_MODEL="/home/gfx/pop-bridge/models/Qwen3.5-2B-Base"
        DEF_SPLIT=14; HIDDEN=2048; KV=512; LAYERS=24; TAIL=512; CTX=512
        ;;
    4b|4B)
        DEF_MODEL="/home/gfx/pop-bridge/models/Qwen3.5-4B"
        DEF_SPLIT=17; HIDDEN=2560; KV=1024; LAYERS=32; TAIL=512; CTX=512
        ;;
    *) die "--size must be 2b or 4b" ;;
esac

[ -n "$MODEL" ]  || MODEL="$DEF_MODEL"
[ -n "$TAG" ]    || TAG="ced_$SIZE"
[ -n "$CORPUS" ] || CORPUS="${CED_CORPUS_INDEX:-/home/gfx/pop-bridge/data/corpus/index.json}"

[ -d "$MODEL" ] || die "model dir not found: $MODEL (use --model)"
[ -f "$CORPUS" ] || die "corpus index not found: $CORPUS (use --corpus-index; generate it with gen_ced_corpus.py)"
mkdir -p "$OUT"

echo "[ced_train] size=$SIZE model=$MODEL"
echo "            hidden=$HIDDEN kv_dim=$KV layers=$LAYERS split=$DEF_SPLIT tail=$TAIL ctx=$CTX"
echo "            corpus=$CORPUS out=$OUT tag=$TAG"

export CED_MODEL_PATH="$MODEL"
export CED_CORPUS_INDEX="$CORPUS"

# a passthrough --split-layer wins over the size default
EXTRA=()
have_split=0
for a in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do [ "$a" = "--split-layer" ] && have_split=1; done
[ "$have_split" = 1 ] || EXTRA+=(--split-layer "$DEF_SPLIT")
have_tail=0
for a in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do [ "$a" = "--tail" ] && have_tail=1; done
[ "$have_tail" = 1 ] || EXTRA+=(--tail "$TAIL")
have_ctx=0
for a in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do [ "$a" = "--ctx" ] && have_ctx=1; done
[ "$have_ctx" = 1 ] || EXTRA+=(--ctx "$CTX")

echo "[ced_train] extra: ${EXTRA[*]}"
set -x
"$PY" "$HERE/ced_final_fixed.py" \
    "${EXTRA[@]}" \
    --out "$OUT" \
    --tag "$TAG" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}

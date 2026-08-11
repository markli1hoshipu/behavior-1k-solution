#!/bin/bash
# Build per-task norm stats + FAST tokenizer for one single-task config.
#
#   scripts/build_task_assets.sh pi_behavior_b1k_task87_ft_nolang
#
# Why per-task rather than one shared asset set: normalisation and the action codebook are
# both fitted to a distribution, and a single task's distribution is much narrower than a
# 5- or 100-task mixture. Fitting them on the task actually being trained means the state
# z-scores use that task's spread, and the BPE merges are learned from that task's own
# action chunks. It also makes the base_qvel correction moot -- stats computed from the
# 2026 demos already carry the robot-frame convention, so there is nothing to patch.
#
# The cost is that models trained under different assets are not comparable on loss value,
# and each checkpoint must carry its own assets to eval correctly. Orbax already copies the
# asset directory into every checkpoint, so that part is handled.
#
# ORDER MATTERS: train_fast_tokenizer loads norm_stats to normalise actions before the DCT.
# Running it first silently trains the codebook on unnormalised actions.

set -euo pipefail

REPO=/work6/markhsp/behavior-1k-solution
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/openpi/src"

CFG="${1:-}"
if [ -z "$CFG" ]; then
    echo "usage: $0 <config-name>   e.g. pi_behavior_b1k_task87_ft_nolang" >&2
    exit 2
fi

# Task index comes from the config name, and is cross-checked against the episode range the
# config actually selects -- a mismatch would silently fit the stats to the wrong task.
TASK=$(sed -E 's/.*_task([0-9]+)_.*/\1/' <<< "$CFG")
if ! [[ "$TASK" =~ ^[0-9]+$ ]]; then
    echo "cannot parse a task index out of '$CFG'" >&2
    exit 2
fi

ASSETS="$REPO/outputs/assets/_pertask/task${TASK}/behavior-1k/2026-challenge-demos"
mkdir -p "$ASSETS"
ln -sfn "_pertask/task${TASK}" "$REPO/outputs/assets/${CFG}"

# Read the FAST parameters off the model config rather than repeating them here: they must
# match what the model expects at train time or the auxiliary head decodes a different
# codebook than the one the targets were built with.
read -r EPS_LO EPS_HI DIMS VOCAB <<< "$(.venv/bin/python3 - "$CFG" <<'PY'
import sys
from b1k.training import config as C
c = C.get_config(sys.argv[1])
ep = c.data.base_config.episodes_index
print(ep[0], ep[-1], c.model.fast_encoded_dims, c.model.fast_vocab_size)
PY
)"

if [ "$((EPS_LO / 200))" != "$TASK" ]; then
    echo "config $CFG selects episodes ${EPS_LO}..${EPS_HI}, which is task $((EPS_LO / 200)), not $TASK" >&2
    exit 1
fi

echo "=== task $TASK | episodes ${EPS_LO}..${EPS_HI} | dims $DIMS | vocab $VOCAB ==="
echo "=== assets -> $ASSETS ==="

echo "--- 1/2 norm stats ---"
.venv/bin/python3 -u scripts/compute_norm_stats.py \
    --config-name "$CFG" --per-timestamp --num-workers 64

echo "--- 2/2 FAST tokenizer ---"
.venv/bin/python3 -u scripts/train_fast_tokenizer.py \
    --config-name "$CFG" --num-workers 64 \
    --encoded-dims "$DIMS" --vocab-size "$VOCAB" --scale 10.0

echo "--- result ---"
ls -la "$ASSETS"
[ -f "$ASSETS/fast_tokenizer/metadata.json" ] && cat "$ASSETS/fast_tokenizer/metadata.json"

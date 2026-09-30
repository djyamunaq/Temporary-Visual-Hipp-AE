#!/usr/bin/env bash
# Data-efficiency sweep: subset sizes x {no attention, attention} for a fixed pooling size.
# A run is skipped if its metrics.json exists (main.py writes it last, so it marks a finished run);
# re-running the script after a crash therefore only redoes what is missing.
# Extra arguments are passed to main.py, e.g. ./run_attention.sh --grid-cells --noise-sigma 0.2
set -euo pipefail

POOL_H=1
POOL_W=1
SIZES=(0.05 0.1 0.25 0.5 1.0)
SEED=0
LOG_DIR=./logs
CKPT_BASE=./ae_model/feature_extractor_ae_checkpoint   # passed to main.py, so skip check and run agree

mkdir -p "$LOG_DIR"

# mirror main.py's mode_tag prefix
mode="features_only"
for a in "$@"; do
    if [[ "$a" == "--grid-cells" ]]; then mode="grid"; fi
done

pool_tag="pool${POOL_H}x${POOL_W}"
log_prefix=""
if [[ "$mode" == "grid" ]]; then log_prefix="grid_"; fi   # keep --grid-cells logs from overwriting features-only logs

for size in "${SIZES[@]}"; do
    for att in "" "--attention"; do
        name="${log_prefix}${pool_tag}_subset${size}${att:+_att}"     # log name (pool tag avoids clobbering the 1x1 logs)
        run_dir="${CKPT_BASE}/${mode}_${pool_tag}${att:+_att}_subset${size}"   # mirrors main.py's checkpoint_dir

        if [[ -f "$run_dir/metrics.json" ]]; then
            echo "=== $name: already done, skipping ==="
            continue
        fi

        echo "=== $name ==="
        python main.py \
            --subset-size "$size" --seed "$SEED" \
            --pool-output-size "$POOL_H" "$POOL_W" \
            --checkpoint-base "$CKPT_BASE/" \
            $att "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    done
done

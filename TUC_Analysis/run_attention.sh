#!/usr/bin/env bash
# Data-efficiency sweep: pooling sizes x subset sizes x {no attention, attention}.
# A run is skipped if its metrics.json exists (main.py writes it last, so it marks a finished run);
# re-running the script after a crash therefore only redoes what is missing.
# Extra arguments are passed to main.py, e.g. ./run_attention.sh --grid-cells --noise-sigma 0.2
set -euo pipefail

POOLS=("1 1" "2 2")            # "H W" pairs for --pool-output-size
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

log_prefix=""
if [[ "$mode" == "grid" ]]; then log_prefix="grid_"; fi   # keep --grid-cells logs from overwriting features-only logs

for pool in "${POOLS[@]}"; do
    read -r pool_h pool_w <<< "$pool"
    pool_tag="pool${pool_h}x${pool_w}"

    for size in "${SIZES[@]}"; do
        for att in "" "--attention"; do
            name="${log_prefix}${pool_tag}_subset${size}${att:+_att}"
            run_dir="${CKPT_BASE}/${mode}_${pool_tag}${att:+_att}_subset${size}"

            if [[ -f "$run_dir/metrics.json" ]]; then
                echo "=== $name: already done, skipping ==="
                continue
            fi

            echo "=== $name ==="
            python main.py \
                --subset-size "$size" --seed "$SEED" \
                --pool-output-size "$pool_h" "$pool_w" \
                --checkpoint-base "$CKPT_BASE/" \
                --n_hidden 100 \
                $att "$@" 2>&1 | tee "$LOG_DIR/$name.log"
        done
    done
done
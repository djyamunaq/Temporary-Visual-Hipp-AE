# Data-efficiency sweep: subset sizes x {no attention, attention}.
# Extra arguments are passed to main.py, e.g. ./run_attention.sh --grid-cells --noise-sigma 0.2
set -euo pipefail

SIZES=(0.05 0.1 0.25 0.5 1.0)
SEED=0
LOG_DIR=./logs
mkdir -p "$LOG_DIR"

for size in "${SIZES[@]}"; do
    for att in "" "--attention"; do
        name="subset${size}${att:+_att}"
        echo "=== $name ==="
        python main.py --subset-size "$size" --seed "$SEED" $att "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    done
done
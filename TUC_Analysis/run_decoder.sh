#!/usr/bin/env bash
# Runs attention_visualisation.py sequentially for paired (checkpoint, n_hidden) settings.
# n_hidden is tied to the checkpoint's encoder, so the two arrays are PAIRED (same index), not a grid.

CHECKPOINTS=(
  "./ae_model/feature_extractor_ae_checkpoint/features_only_pool1x1_att_subset1.0/"
  "./ae_model/feature_extractor_ae_checkpoint/features_only_pool1x1_subset0.1/"
  "./ae_model/feature_extractor_ae_checkpoint/features_only_pool1x1_att_subset0.1/"
  "./ae_model/feature_extractor_ae_checkpoint_n200/features_only_pool1x1_att_subset1.0/"
  "./ae_model/feature_extractor_ae_checkpoint_n200/features_only_pool1x1_subset1.0/"
  "./ae_model/feature_extractor_ae_checkpoint_n200/features_only_pool2x2_att_subset1.0/"
  "./ae_model/feature_extractor_ae_checkpoint_n200/features_only_pool2x2_subset1.0/"
)
N_HIDDEN=(100 100 100 200 200 200 200)

TRAIN_DECODER=true     # set to false to load an existing decoder_model.pt
BATCH_SIZE=128
DEVICE=cuda
COOLDOWN=60            # seconds to pause between runs (GPU/CPU cooldown)

if [ "${#CHECKPOINTS[@]}" -ne "${#N_HIDDEN[@]}" ]; then
  echo "CHECKPOINTS and N_HIDDEN must have the same length" >&2
  exit 1
fi

TRAIN_FLAG=""
[ "$TRAIN_DECODER" = true ] && TRAIN_FLAG="--train-decoder"

failed=()
for i in "${!CHECKPOINTS[@]}"; do
  ckpt="${CHECKPOINTS[$i]}"
  nh="${N_HIDDEN[$i]}"
  echo "=== [$((i+1))/${#CHECKPOINTS[@]}] checkpoint=$ckpt n_hidden=$nh ==="
  if ! python attention_visualisation.py \
        --checkpoint-path "$ckpt" \
        --n-hidden "$nh" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        $TRAIN_FLAG; then
    echo "!!! FAILED: $ckpt (n_hidden=$nh)" >&2
    failed+=("$ckpt")
  fi

  # cooldown between runs (skipped after the last one)
  if [ "$i" -lt "$((${#CHECKPOINTS[@]}-1))" ]; then
    echo "Cooling down for ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "Failed runs:"; printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All runs finished."
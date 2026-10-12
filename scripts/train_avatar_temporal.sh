#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <avatar_name>" >&2
  exit 2
fi

avatar_name="$1"
dataset_dir="data/${avatar_name}"
training_video="${dataset_dir}/train.mp4"
checkpoint_dir="${CHECKPOINT_DIR:-checkpoints/${avatar_name}}"
preview_dir="${PREVIEW_DIR:-outputs/train_preview_${avatar_name}}"
feather_hubert_checkpoint="assets/featherhubert/feather_hubert_188_latest_99.pth"

# Temporal-mouth training defaults.
# One temporal sample contains two adjacent frames, so batch 8 is roughly
# comparable to batch 16 in the old single-frame trainer in terms of images/step.
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"

# Conservative defaults from train_mouth_roi_temporal_loss.py.
MOUTH_WEIGHT="${MOUTH_WEIGHT:-4.0}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.5}"
TEMPORAL_MOUTH_WEIGHT="${TEMPORAL_MOUTH_WEIGHT:-4.0}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.01}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-1}"

# Keep intermediate checkpoints by default so different epochs can be compared.
KEEP_ONLY_LAST="${KEEP_ONLY_LAST:-0}"

if [ ! -f "$training_video" ]; then
  echo "Error: training video does not exist: ${training_video}" >&2
  exit 1
fi

if [ ! -f "train_mouth_roi_temporal_loss.py" ]; then
  echo "Error: train_mouth_roi_temporal_loss.py not found. Run from the FeatherTalk repository root." >&2
  exit 1
fi

echo "[1/5] Preprocessing avatar data: ${training_video}"
python data_utils/process.py \
  "$training_video" \
  --asr hubert \
  --feather_hubert_checkpoint \
  "$feather_hubert_checkpoint"

echo "[2/5] Validating video/audio feature alignment"
python - "$dataset_dir" <<'PY'
import os
import sys
import numpy as np

dataset_dir = sys.argv[1]
img_dir = os.path.join(dataset_dir, "full_body_img")
aud_path = os.path.join(dataset_dir, "aud_hu.npy")

n_frames = sum(name.lower().endswith(".jpg") for name in os.listdir(img_dir))
aud = np.load(aud_path, mmap_mode="r")
n_audio = int(aud.shape[0])
delta = n_audio - n_frames

print(f"Video frames:         {n_frames}")
print(f"Audio feature frames: {n_audio}")
print(f"Difference:           {delta:+d}")

if n_audio < n_frames:
    raise SystemExit(
        f"ERROR: audio features are shorter than video by {-delta} frames. "
        "Do not train until audio/video alignment is fixed."
    )
if delta > 3:
    raise SystemExit(
        f"ERROR: audio features are longer than video by {delta} frames. "
        "This may indicate the old chunked-AAC timing problem."
    )
print("Alignment check passed.")
PY

echo "[3/5] Preparing output directories"
mkdir -p "$checkpoint_dir"
mkdir -p "$preview_dir"

echo "[4/5] Training with mouth ROI + adjacent-frame temporal loss"
echo "      epochs=$EPOCHS batch=$BATCH_SIZE"
echo "      mouth=$MOUTH_WEIGHT temporal=$TEMPORAL_WEIGHT temporal_mouth=$TEMPORAL_MOUTH_WEIGHT"
echo "      checkpoints=$checkpoint_dir"

python train_mouth_roi_temporal_loss.py \
  --dataset_dir "$dataset_dir" \
  --save_dir "$checkpoint_dir" \
  --asr hubert \
  --unet original \
  --epochs "$EPOCHS" \
  --batchsize "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --save_every "$SAVE_EVERY" \
  --mouth_weight "$MOUTH_WEIGHT" \
  --temporal_weight "$TEMPORAL_WEIGHT" \
  --temporal_mouth_weight "$TEMPORAL_MOUTH_WEIGHT" \
  --perceptual_weight "$PERCEPTUAL_WEIGHT" \
  --temporal_stride "$TEMPORAL_STRIDE" \
  --see_res \
  --see_res_dir "$preview_dir"

echo "[5/5] Checkpoint handling"
if [ "$KEEP_ONLY_LAST" = "1" ]; then
  echo "Keeping only last.pth"
  find "$checkpoint_dir" -maxdepth 1 -type f -name "*.pth" ! -name "last.pth" -delete
else
  echo "Keeping intermediate checkpoints for visual/inference comparison."
fi

echo "Done."
echo "Checkpoint directory: $checkpoint_dir"
echo "Preview directory:    $preview_dir"

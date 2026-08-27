#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <avatar_name>" >&2
  exit 2
fi

avatar_name="$1"
dataset_dir="data/${avatar_name}"
training_video="${dataset_dir}/train.mp4"
checkpoint_dir="checkpoints/${avatar_name}"
preview_dir="outputs/train_preview"
feather_hubert_checkpoint="assets/featherhubert/feather_hubert_188_latest_99.pth"

if [ ! -f "$training_video" ]; then
  echo "Error: training video does not exist: ${training_video}" >&2
  exit 1
fi

echo "[1/3] Preprocessing avatar data: ${training_video}"
python data_utils/process.py \
  "$training_video" \
  --asr hubert \
  --feather_hubert_checkpoint \
  "$feather_hubert_checkpoint"

echo "[2/3] Preparing output directories"
mkdir -p "$checkpoint_dir"
mkdir -p "$preview_dir"

echo "[3/3] Training avatar: ${avatar_name}"
python train_mouth_roi_temporal_loss.py \
  --dataset_dir "$dataset_dir" \
  --save_dir "$checkpoint_dir" \
  --asr hubert \
  --unet original \
  --epochs 200 \
  --batchsize 16 \
  --num_workers 4 \
  --save_every 5 \
  --see_res \
  --see_res_dir "$preview_dir"

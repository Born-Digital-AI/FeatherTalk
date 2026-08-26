"""Train UNet with mouth temporal losses that emphasize silent segments."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset_mouth_roi import MouthRoiConfig, TemporalMouthRoiDataset
from model_factory import UNET_CHOICES, create_model
from train import PerceptualLoss, get_training_device, resume_if_any, save_checkpoint
from train_mouth_roi_temporal_loss import _to_img, dump_sample


@dataclass(frozen=True)
class SilenceLossWeights:
    mouth: float = 4.0
    temporal: float = 0.5
    temporal_mouth: float = 4.0
    perceptual: float = 0.01
    silence_mouth: float = 8.0
    silence_temporal: float = 1.0
    silence_temporal_mouth: float = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original UNet with silence-aware mouth ROI and temporal losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--asr", type=str, default="hubert", choices=["wenet", "hubert"])
    parser.add_argument("--unet", type=str, default="original", choices=UNET_CHOICES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--see_res", action="store_true")
    parser.add_argument("--see_res_dir", type=str, default="./train_tmp_img_silence_mouth_temporal")

    parser.add_argument("--mouth_weight", type=float, default=4.0)
    parser.add_argument("--temporal_weight", type=float, default=0.5)
    parser.add_argument("--temporal_mouth_weight", type=float, default=4.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument("--temporal_stride", type=int, default=1)

    parser.add_argument("--silence_sample_ratio", type=float, default=0.4)
    parser.add_argument("--silence_threshold_percentile", type=float, default=25.0)
    parser.add_argument("--silence_audio_half_window", type=int, default=4)
    parser.add_argument("--silence_min_run", type=int, default=2)
    parser.add_argument("--silence_mouth_weight", type=float, default=8.0)
    parser.add_argument("--silence_temporal_weight", type=float, default=1.0)
    parser.add_argument("--silence_temporal_mouth_weight", type=float, default=10.0)
    parser.add_argument("--sampler_seed", type=int, default=0)

    parser.add_argument("--mouth_start", type=int, default=90)
    parser.add_argument("--mouth_end", type=int, default=110)
    parser.add_argument("--mouth_expand_x", type=float, default=1.45)
    parser.add_argument("--mouth_expand_y", type=float, default=1.75)
    parser.add_argument("--mouth_min_w", type=int, default=52)
    parser.add_argument("--mouth_min_h", type=int, default=36)
    return parser.parse_args()


def audio_window_energy(features: np.ndarray, half_window: int = 4) -> np.ndarray:
    if half_window < 0:
        raise ValueError("half_window must be >= 0")
    frame_energy = np.mean(np.abs(features), axis=tuple(range(1, features.ndim)))
    energies = np.empty(features.shape[0], dtype=np.float32)
    for idx in range(features.shape[0]):
        left = max(0, idx - half_window)
        right = min(features.shape[0], idx + half_window + 1)
        energies[idx] = float(frame_energy[left:right].mean())
    return energies


def _keep_min_runs(flags: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return flags
    kept = np.zeros_like(flags, dtype=bool)
    start = None
    for idx, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            if idx - start >= min_run:
                kept[start:idx] = True
            start = None
    return kept


def detect_silence_frames(
    features: np.ndarray,
    percentile: float = 25.0,
    half_window: int = 4,
    min_run: int = 1,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if percentile <= 0.0:
        return np.zeros((features.shape[0],), dtype=bool)
    if percentile > 100.0:
        raise ValueError("percentile must be <= 100")
    energies = audio_window_energy(features, half_window=half_window)
    threshold = float(np.percentile(energies, percentile))
    flags = energies <= threshold
    return _keep_min_runs(flags, min_run=min_run)


def silent_pair_indices(silent_frames: np.ndarray, dataset_len: int, temporal_stride: int) -> list[int]:
    indices = []
    for idx in range(dataset_len):
        if idx + temporal_stride < len(silent_frames) and silent_frames[idx] and silent_frames[idx + temporal_stride]:
            indices.append(idx)
    return indices


def _percent(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(part) / float(total)


def rate_silent_pair_coverage(silent_pair_percent: float) -> str:
    if silent_pair_percent < 3.0:
        return "too low"
    if silent_pair_percent < 8.0:
        return "usable but weak"
    if silent_pair_percent < 20.0:
        return "good"
    if silent_pair_percent <= 40.0:
        return "very good"
    return "high; watch speaking quality"


def format_silence_training_summary(
    silent_frames: int,
    total_frames: int,
    silent_pairs: int,
    total_pairs: int,
    target_sample_ratio: float,
) -> str:
    frame_percent = _percent(silent_frames, total_frames)
    pair_percent = _percent(silent_pairs, total_pairs)
    rating = rate_silent_pair_coverage(pair_percent)
    return (
        "Silence-aware training: "
        f"{silent_frames}/{total_frames} silent frames ({frame_percent:.1f}%), "
        f"{silent_pairs}/{total_pairs} silent temporal pairs ({pair_percent:.1f}% - {rating}), "
        f"target sample ratio={target_sample_ratio * 100.0:.1f}%"
    )


class SilenceAwareTemporalSampler(Sampler[int]):
    def __init__(
        self,
        dataset_len: int,
        silent_pair_indices: list[int],
        silence_sample_ratio: float = 0.4,
        seed: int = 0,
    ):
        if dataset_len < 0:
            raise ValueError("dataset_len must be >= 0")
        if not 0.0 <= silence_sample_ratio <= 1.0:
            raise ValueError("silence_sample_ratio must be in [0, 1]")
        self.dataset_len = dataset_len
        self.silent_pair_indices = list(silent_pair_indices)
        self.silence_sample_ratio = silence_sample_ratio
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.dataset_len == 0:
            return iter(())
        all_indices = list(range(self.dataset_len))
        if not self.silent_pair_indices:
            rng.shuffle(all_indices)
            return iter(all_indices)

        n_silent = int(round(self.dataset_len * self.silence_sample_ratio))
        n_speech = self.dataset_len - n_silent
        speech_pool = [idx for idx in all_indices if idx not in set(self.silent_pair_indices)]
        if not speech_pool:
            speech_pool = all_indices

        sampled = [
            self.silent_pair_indices[rng.randrange(len(self.silent_pair_indices))]
            for _ in range(n_silent)
        ]
        sampled.extend(speech_pool[rng.randrange(len(speech_pool))] for _ in range(n_speech))
        rng.shuffle(sampled)
        return iter(sampled)

    def __len__(self) -> int:
        return self.dataset_len


class SilenceAwareTemporalMouthRoiDataset(TemporalMouthRoiDataset):
    def __init__(
        self,
        dataset_dir: str,
        mode: str,
        mouth_config: MouthRoiConfig | None = None,
        temporal_stride: int = 1,
        silence_threshold_percentile: float = 25.0,
        silence_audio_half_window: int = 4,
        silence_min_run: int = 2,
    ):
        super().__init__(
            dataset_dir,
            mode,
            mouth_config=mouth_config,
            temporal_stride=temporal_stride,
        )
        self.silent_frames = detect_silence_frames(
            self.audio_feats,
            percentile=silence_threshold_percentile,
            half_window=silence_audio_half_window,
            min_run=silence_min_run,
        )
        self.silent_pair_indices = silent_pair_indices(
            self.silent_frames,
            len(self),
            self.temporal_stride,
        )
        self._silent_pair_index_set = set(self.silent_pair_indices)

    def is_silent_pair(self, idx: int) -> bool:
        return idx in self._silent_pair_index_set

    def __getitem__(self, idx: int):
        items = super().__getitem__(idx)
        return (*items, torch.tensor(self.is_silent_pair(idx), dtype=torch.bool))


def _per_frame_mouth_loss(preds: torch.Tensor, labels: torch.Tensor, mouth_masks: torch.Tensor) -> torch.Tensor:
    masks = mouth_masks.to(dtype=preds.dtype)
    diff = (preds - labels).abs() * masks
    denom = masks.flatten(1).sum(dim=1).clamp_min(1.0) * preds.shape[1]
    return diff.flatten(1).sum(dim=1) / denom


def _per_pair_temporal_losses(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_delta = preds[:, 1] - preds[:, 0]
    label_delta = labels[:, 1] - labels[:, 0]
    union_mask = mouth_masks.max(dim=1).values
    full = (pred_delta - label_delta).abs().flatten(1).mean(dim=1)
    mouth = _per_frame_mouth_loss(pred_delta, label_delta, union_mask)
    return full, mouth


def compute_silence_aware_total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
    silence_flags: torch.Tensor,
    pixel_criterion: nn.Module,
    perceptual_loss: nn.Module,
    weights: SilenceLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, pair_len = preds.shape[:2]
    flat_preds = preds.reshape(batch_size * pair_len, *preds.shape[2:])
    flat_labels = labels.reshape(batch_size * pair_len, *labels.shape[2:])
    flat_masks = mouth_masks.reshape(batch_size * pair_len, *mouth_masks.shape[2:])

    silence_flags = silence_flags.to(device=preds.device, dtype=torch.bool)
    pair_weights = torch.where(
        silence_flags,
        torch.full_like(silence_flags, weights.silence_mouth, dtype=preds.dtype),
        torch.full_like(silence_flags, weights.mouth, dtype=preds.dtype),
    )
    temporal_weights = torch.where(
        silence_flags,
        torch.full_like(silence_flags, weights.silence_temporal, dtype=preds.dtype),
        torch.full_like(silence_flags, weights.temporal, dtype=preds.dtype),
    )
    temporal_mouth_weights = torch.where(
        silence_flags,
        torch.full_like(silence_flags, weights.silence_temporal_mouth, dtype=preds.dtype),
        torch.full_like(silence_flags, weights.temporal_mouth, dtype=preds.dtype),
    )

    loss_pixel = pixel_criterion(flat_preds, flat_labels)
    per_frame_mouth = _per_frame_mouth_loss(flat_preds, flat_labels, flat_masks)
    frame_weights = pair_weights.repeat_interleave(pair_len)
    loss_mouth = per_frame_mouth.mean()
    loss_mouth_weighted = (per_frame_mouth * frame_weights).mean()

    per_pair_temporal, per_pair_temporal_mouth = _per_pair_temporal_losses(preds, labels, mouth_masks)
    loss_temporal = per_pair_temporal.mean()
    loss_temporal_mouth = per_pair_temporal_mouth.mean()
    loss_temporal_weighted = (per_pair_temporal * temporal_weights).mean()
    loss_temporal_mouth_weighted = (per_pair_temporal_mouth * temporal_mouth_weights).mean()

    loss_perceptual = perceptual_loss(flat_preds, flat_labels)
    total = (
        loss_pixel
        + loss_mouth_weighted
        + loss_temporal_weighted
        + loss_temporal_mouth_weighted
        + weights.perceptual * loss_perceptual
    )
    return total, {
        "full": float(loss_pixel.detach().cpu()),
        "mouth": float(loss_mouth.detach().cpu()),
        "mouth_w": float(loss_mouth_weighted.detach().cpu()),
        "temp": float(loss_temporal.detach().cpu()),
        "temp_w": float(loss_temporal_weighted.detach().cpu()),
        "temp_mouth": float(loss_temporal_mouth.detach().cpu()),
        "temp_mouth_w": float(loss_temporal_mouth_weighted.detach().cpu()),
        "percep": float(loss_perceptual.detach().cpu()),
        "silence_ratio": float(silence_flags.float().mean().detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
    device: torch.device,
    progress_desc: str,
    dataset_len: int,
    weights: SilenceLossWeights,
):
    net.train()
    with tqdm(total=dataset_len * 2, desc=progress_desc, unit="frame") as progress:
        for imgs, labels, audio_feat, mouth_masks, silence_flags in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)
            mouth_masks = mouth_masks.to(device)
            silence_flags = silence_flags.to(device)

            batch_size, pair_len = imgs.shape[:2]
            flat_imgs = imgs.reshape(batch_size * pair_len, *imgs.shape[2:])
            flat_audio = audio_feat.reshape(batch_size * pair_len, *audio_feat.shape[2:])

            flat_preds = net(flat_imgs, flat_audio)
            preds = flat_preds.reshape(batch_size, pair_len, *flat_preds.shape[1:])

            loss, parts = compute_silence_aware_total_loss(
                preds,
                labels,
                mouth_masks,
                silence_flags,
                pixel_criterion,
                perceptual_loss,
                weights,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            progress.set_postfix(
                {
                    "loss": parts["total"],
                    "mouth": parts["mouth_w"],
                    "temp_m": parts["temp_mouth_w"],
                    "sil": parts["silence_ratio"],
                }
            )
            progress.update(batch_size * pair_len)


def dump_silence_sample(
    net: nn.Module,
    dataset: SilenceAwareTemporalMouthRoiDataset,
    save_dir: str,
    epoch: int,
    device: torch.device,
):
    if dataset.silent_pair_indices:
        idx = random.choice(dataset.silent_pair_indices)
    else:
        dump_sample(net, dataset, save_dir, epoch, device)
        return
    imgs, labels, audio_feat, mouth_masks, _ = dataset[idx]
    with torch.no_grad():
        preds = net(imgs.to(device), audio_feat.to(device))
    panels = [
        _to_img(preds[0]),
        _to_img(labels[0]),
        _to_img(preds[1]),
        _to_img(labels[1]),
    ]
    cv2.imwrite(
        os.path.join(save_dir, f"epoch_{epoch}_silent_pred0_real0_pred1_real1.jpg"),
        np.concatenate(panels, axis=1),
    )
    mask_img = (mouth_masks.max(dim=0).values.numpy()[0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_silent_mouth_union_mask.jpg"), mask_img)


def main():
    args = parse_args()
    device = get_training_device(args.allow_cpu)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.see_res:
        os.makedirs(args.see_res_dir, exist_ok=True)

    mouth_config = MouthRoiConfig(
        start=args.mouth_start,
        end=args.mouth_end,
        expand_x=args.mouth_expand_x,
        expand_y=args.mouth_expand_y,
        min_w=args.mouth_min_w,
        min_h=args.mouth_min_h,
    )
    dataset = SilenceAwareTemporalMouthRoiDataset(
        args.dataset_dir,
        args.asr,
        mouth_config=mouth_config,
        temporal_stride=args.temporal_stride,
        silence_threshold_percentile=args.silence_threshold_percentile,
        silence_audio_half_window=args.silence_audio_half_window,
        silence_min_run=args.silence_min_run,
    )
    sampler = SilenceAwareTemporalSampler(
        dataset_len=len(dataset),
        silent_pair_indices=dataset.silent_pair_indices,
        silence_sample_ratio=args.silence_sample_ratio,
        seed=args.sampler_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(format_silence_training_summary(
        silent_frames=int(dataset.silent_frames.sum()),
        total_frames=len(dataset.silent_frames),
        silent_pairs=len(dataset.silent_pair_indices),
        total_pairs=len(dataset),
        target_sample_ratio=args.silence_sample_ratio,
    ))

    net = create_model(args.asr, args.unet).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pixel_criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss(nn.MSELoss(), device=device)
    weights = SilenceLossWeights(
        mouth=args.mouth_weight,
        temporal=args.temporal_weight,
        temporal_mouth=args.temporal_mouth_weight,
        perceptual=args.perceptual_weight,
        silence_mouth=args.silence_mouth_weight,
        silence_temporal=args.silence_temporal_weight,
        silence_temporal_mouth=args.silence_temporal_mouth_weight,
    )

    start_epoch = resume_if_any(args.resume, net, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            net,
            loader,
            optimizer,
            pixel_criterion,
            perceptual_loss,
            device,
            progress_desc=f"Epoch {epoch + 1}/{args.epochs}",
            dataset_len=len(dataset),
            weights=weights,
        )

        is_save_epoch = (epoch + 1) % args.save_every == 0
        if is_save_epoch or epoch == args.epochs - 1:
            save_checkpoint(os.path.join(args.save_dir, f"{epoch}.pth"), net, optimizer, epoch)
            save_checkpoint(os.path.join(args.save_dir, "last.pth"), net, optimizer, epoch)

        if args.see_res:
            dump_silence_sample(net, dataset, args.see_res_dir, epoch, device)


if __name__ == "__main__":
    main()

"""Measure production FP32 inference/PNG patches and optional batch parity."""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--audio", type=Path, required=True, help="Mono PCM16 WAV; resampled to 16 kHz if needed")
    parser.add_argument("--frames", type=int, default=125)
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import app
    from dihuman_run import DiHumanProcessor

    if args.avatar not in app.list_avatars():
        parser.error("Unknown avatar")
    app.preload_onnxruntime_cuda()
    unet = app.ensure_streaming_unet(args.avatar)
    processor = DiHumanProcessor(
        str(app.DATA_DIR / args.avatar), asr="hubert", unet_onnx=str(unet),
        feather_hubert_checkpoint=str(app.FEATHER_HUBERT_CHECKPOINT), include_last_frame=True,
    )
    if args.batch > 1 and not processor.supports_realtime_batch():
        parser.error("This ONNX has fixed batch=1. Re-export with pth2onnx.py before benchmarking batches.")
    processor.enable_realtime_io_binding()
    with wave.open(str(args.audio), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            parser.error("Audio must be mono PCM16 WAV")
        pcm = app._resample_pcm16_bytes(audio.readframes(audio.getnframes()), audio.getframerate())
    pcm = np.frombuffer(pcm, np.int16)
    started = time.perf_counter()
    windows = processor.prepare_offline_features(pcm)
    encoder_ms = (time.perf_counter() - started) * 1000
    count = min(args.frames, len(windows))
    if not count:
        parser.error("Audio contains no frames")

    # Warm the actual execution path before timing; retain the same weights and FP32.
    processor.process_realtime(np.zeros(640, np.int16), windows[0])
    picker = processor.frame_picker
    positions = []
    for _ in range(count):
        positions.append(picker.index)
        picker.advance()

    baseline = []
    render_ms, encode_ms, sizes = [], [], []
    for index, features in zip(positions, windows):
        picker.index = index
        started = time.perf_counter()
        image = processor.process_realtime(np.zeros(640, np.int16), features)
        render_ms.append((time.perf_counter() - started) * 1000)
        x1, y1, x2, y2 = processor.bbox_list[index]
        patch = image[y1:y2, x1:x2].copy()
        baseline.append(patch)
        started = time.perf_counter()
        encoded = app._frame_data_url(patch)
        encode_ms.append((time.perf_counter() - started) * 1000)
        sizes.append(len(encoded))

    report = {
        "unet_providers": processor.ort_unet.get_providers(),
        "encoder_device": str(processor.feather_device),
        "io_binding": bool(processor.realtime_io_binding), "frames": count,
        "encoder_ms_per_reply": round(encoder_ms, 2),
        "render_mean_ms": round(float(np.mean(render_ms)), 2),
        "render_p95_ms": round(float(np.percentile(render_ms, 95)), 2),
        "patch_encode_mean_ms": round(float(np.mean(encode_ms)), 2),
        "render_plus_encode_fps": round(1000 / float(np.mean(render_ms) + np.mean(encode_ms)), 2),
        "patch_mean_kib": round(float(np.mean(sizes)) / 1024, 2),
        "patch_mbit_s_at_25fps": round(float(np.mean(sizes)) * 25 * 8 / 1e6, 2),
        "excludes": "network, background cache misses, browser decoding and idle preparation",
    }
    if args.batch > 1:
        batch_times = []
        maximum_error = 0
        changed_pixels, total_pixels = 0, 0
        for start in range(0, count, args.batch):
            frames = list(zip(positions[start:start + args.batch], windows[start:start + args.batch]))
            began = time.perf_counter()
            images = processor.process_realtime_batch(frames)
            batch_times.append((time.perf_counter() - began) * 1000 / len(frames))
            for offset, ((index, _), image) in enumerate(zip(frames, images)):
                x1, y1, x2, y2 = processor.bbox_list[index]
                difference = np.abs(image[y1:y2, x1:x2].astype(np.int16) - baseline[start + offset].astype(np.int16))
                maximum_error = max(maximum_error, int(difference.max()))
                changed_pixels += int(np.count_nonzero(difference))
                total_pixels += difference.size
        report.update(batch=args.batch, batch_render_mean_ms=round(float(np.mean(batch_times)), 2),
                      batch_max_pixel_error=maximum_error,
                      batch_changed_channel_fraction=changed_pixels / total_pixels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

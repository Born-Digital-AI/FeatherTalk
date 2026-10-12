"""Streaming inference demo for FeatherTalk.

This file keeps the original Wenet ONNX streaming path and adds a HuBERT-shape
path for FeatherHuBERT.  The two modes differ only on the audio encoder side
and the UNet audio input shape:

* wenet: fbank -> Wenet encoder ONNX -> audio shape [1, 128, 16, 32]
* feather_hubert: waveform -> FeatherHuBERT -> audio shape [1, 16, 32, 32]

Input audio is still consumed as 10 ms / 16 kHz PCM chunks.  The Python
FeatherHuBERT path recomputes features from the current utterance buffer, so it
is meant as a correctness/reference streaming demo.  For production/mobile use,
export FeatherHuBERT to ONNX and pass it through --encoder_onnx.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Sequence

import cv2
import numpy as np
import onnxruntime
from frame_store import ImageFrames
from face_utils import (
    FACE_BORDER,
    FACE_INNER_SIZE,
    compute_face_bbox,
    crop_face,
    extract_inner,
    mask_mouth,
    read_landmarks,
)

try:
    import kaldi_native_fbank as knf
except ImportError:  # FeatherHuBERT mode does not need fbank.
    knf = None


SAMPLE_RATE = 16000
FRAME_LEN = 160  # 10 ms @ 16 kHz

ASR_WENET = "wenet"
ASR_FEATHER = "feather_hubert"
ASR_CHOICES = (ASR_WENET, ASR_FEATHER, "hubert")
ASR_ALIASES = {"hubert": ASR_FEATHER}

FPS_BY_ASR = {
    ASR_WENET: 20,
    ASR_FEATHER: 25,
}
IDLE_LOOP_BY_ASR = {
    ASR_WENET: 5,
    ASR_FEATHER: 4,
}

WENET_TRIGGER_LEN = 11040  # 690 ms
WENET_CHUNK_DROP = 800  # 50 ms
MEL_BINS = 80
WENET_FEAT_FRAMES = 67
PRE_AUDIO_LEN = 32 * FRAME_LEN  # 320 ms silence prefix for Wenet encoder
PLAY_PRE_PAD = 13440

FEATHER_TOKENS_PER_VIDEO_FRAME = 2
FEATHER_DEFAULT_OUTPUT_DIM = 1024
FEATHER_RIGHT_CONTEXT_FRAMES = 4
FEATHER_STREAM_WINDOW_SAMPLES = 720
FEATHER_STREAM_HOP_SAMPLES = 640
FEATHER_REALTIME_HISTORY_SAMPLES = 2 * SAMPLE_RATE

SILENCE_THRESHOLD = 100
UNET_FEAT_WINDOW = 8
USING_FEAT_INIT = 4

_FBANK_OPTS = None


def parse_args():
    parser = argparse.ArgumentParser(description="FeatherTalk streaming inference demo")
    parser.add_argument("--data_path", type=str, default="./dataset_kanghui_wenet/111/",
                        help="Dataset dir. Supports img_inference/lms_inference or full_body_img/landmarks.")
    parser.add_argument("--audio_wav", type=str, default="1.wav", help="Input test wav")
    parser.add_argument("--out_video", type=str, default="./test_video.mp4")
    parser.add_argument("--out_audio", type=str, default="./test_audio.wav")
    parser.add_argument("--video_size", type=int, nargs=2, default=[1280, 720],
                        help="Output video resolution: W H")
    parser.add_argument("--asr", type=str, default=ASR_WENET, choices=ASR_CHOICES,
                        help="wenet, or feather_hubert. hubert is an alias of feather_hubert here.")
    parser.add_argument("--unet_onnx", type=str, default="",
                        help="UNet ONNX path. Defaults to <data_path>/unet.onnx")
    parser.add_argument("--encoder_onnx", type=str, default="",
                        help="Wenet encoder ONNX, or optional FeatherHuBERT ONNX.")
    parser.add_argument("--feather_hubert_checkpoint", type=str, default="",
                        help="FeatherHuBERT .pth checkpoint when --asr feather_hubert and no --encoder_onnx is used.")
    parser.add_argument("--feather_right_context_frames", type=int, default=FEATHER_RIGHT_CONTEXT_FRAMES,
                        help="Delay FeatherHuBERT feature emission by N video frames for steadier streaming features.")
    parser.add_argument("--fps", type=float, default=0,
                        help="Override output fps. Defaults to 20 for Wenet and 25 for FeatherHuBERT.")
    parser.add_argument("--cpu_onnx", action="store_true",
                        help="Force ONNX Runtime CPUExecutionProvider.")
    return parser.parse_args()


def _normalize_asr(asr: str) -> str:
    return ASR_ALIASES.get(asr, asr)


def _get_fbank_opts():
    global _FBANK_OPTS
    if knf is None:
        raise RuntimeError(
            "kaldi_native_fbank is required for --asr wenet. "
            "Install it or use --asr feather_hubert."
        )
    if _FBANK_OPTS is None:
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = MEL_BINS
        opts.mel_opts.debug_mel = False
        _FBANK_OPTS = opts
    return _FBANK_OPTS


def _ort_providers(force_cpu: bool = False) -> List[str]:
    if force_cpu:
        return ["CPUExecutionProvider"]
    available = set(onnxruntime.get_available_providers())
    providers: List[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _read_landmarks_to_bbox(lms_path: str) -> Tuple[int, int, int, int]:
    return compute_face_bbox(read_landmarks(lms_path))


def _build_unet_inputs(crop_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    crop_ori = crop_img.copy()
    inner = extract_inner(crop_img)
    masked = mask_mouth(inner.copy())

    masked = masked.transpose(2, 0, 1).astype(np.float32) / 255.0
    inner = inner.transpose(2, 0, 1).astype(np.float32) / 255.0
    onnx_in = np.concatenate(
        (np.expand_dims(inner, 0), np.expand_dims(masked, 0)),
        axis=1,
    )
    return onnx_in, crop_ori


def _numeric_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def _sorted_files(directory: str, suffix: str) -> List[Path]:
    root = Path(directory)
    return sorted(
        [path for path in root.iterdir() if path.suffix.lower() == suffix],
        key=_numeric_sort_key,
    )


def _resolve_asset_dirs(data_path: str) -> Tuple[str, str]:
    candidates = [
        ("img_inference", "lms_inference"),
        ("full_body_img", "landmarks"),
    ]
    for img_name, lms_name in candidates:
        img_dir = os.path.join(data_path, img_name)
        lms_dir = os.path.join(data_path, lms_name)
        if os.path.isdir(img_dir) and os.path.isdir(lms_dir):
            return img_dir, lms_dir
    raise FileNotFoundError(
        f"{data_path} must contain img_inference/lms_inference or full_body_img/landmarks."
    )


def _resolve_default_path(path: str, data_path: str, name: str) -> str:
    return path or os.path.join(data_path, name)


class _BounceIndex:
    """0,1,...,N-1,N-2,...,1,0,... index generator."""

    def __init__(self, n_frames: int):
        if n_frames < 2:
            raise ValueError("At least 2 frames are required for streaming inference.")
        self.n_frames = n_frames
        self.index = 0
        self.step = 1

    def advance(self):
        self.index += self.step
        if self.index >= self.n_frames - 1:
            self.step = -1
        elif self.index <= 0:
            self.step = 1


class DiHumanProcessor:
    def __init__(
        self,
        data_path: str,
        asr: str = ASR_WENET,
        unet_onnx: str = "",
        encoder_onnx: str = "",
        feather_hubert_checkpoint: str = "",
        feather_right_context_frames: int = FEATHER_RIGHT_CONTEXT_FRAMES,
        force_cpu_onnx: bool = False,
        max_frames: Optional[int] = None,
        include_last_frame: bool = False,
        defer_models: bool = False,
        lazy_assets: bool = False,
    ):
        self.data_path = data_path
        self.asr = _normalize_asr(asr)
        self.idle_loop = IDLE_LOOP_BY_ASR[self.asr]
        self.providers = _ort_providers(force_cpu_onnx)
        self.feather_right_context_frames = max(0, int(feather_right_context_frames))

        full_body_img_dir, lms_dir = _resolve_asset_dirs(data_path)
        self.full_body_img_list, self.bbox_list = self._load_assets(
            full_body_img_dir, lms_dir, max_frames, include_last_frame, lazy_assets
        )
        self.frame_picker = _BounceIndex(len(self.bbox_list))

        self._unet_path = _resolve_default_path(unet_onnx, data_path, "unet.onnx")
        self._encoder_onnx = encoder_onnx
        self._feather_checkpoint = feather_hubert_checkpoint
        self._models_initialized = False
        self.ort_unet = None

        self.ort_ae: Optional[onnxruntime.InferenceSession] = None
        self.feather_model: Any = None
        self.feather_device: Any = None
        self.feather_output_dim = FEATHER_DEFAULT_OUTPUT_DIM
        self._torch = None
        self._feather_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._feather_expected_frames: Optional[Callable[[int], int]] = None
        self._feather_make_even: Optional[Callable[[Any], Any]] = None

        if not defer_models:
            self.initialize_models()

        self.counter = 0
        self.empty_audio_counter = 56
        self.is_processing = False
        self.silence = True
        self.last_frame_index = 0
        self._reset_audio_state()

    def initialize_models(self) -> None:
        """Load encoder and UNet sessions once for speech or inferred idle."""
        if self._models_initialized:
            return
        if not os.path.exists(self._unet_path):
            raise FileNotFoundError(f"UNet ONNX not found: {self._unet_path}")
        self.ort_unet = onnxruntime.InferenceSession(self._unet_path, providers=self.providers)
        if self.asr == ASR_WENET:
            self._init_wenet_encoder(_resolve_default_path(self._encoder_onnx, self.data_path, "encoder.onnx"))
        else:
            self._init_feather_encoder(self._encoder_onnx, self._feather_checkpoint)
        self._models_initialized = True

    def _load_assets(
        self,
        img_dir: str,
        lms_dir: str,
        max_frames: Optional[int] = None,
        include_last_frame: bool = False,
        lazy_assets: bool = False,
    ) -> Tuple[Sequence[np.ndarray], List[Tuple[int, int, int, int]]]:
        img_paths = _sorted_files(img_dir, ".jpg")
        if not img_paths:
            img_paths = _sorted_files(img_dir, ".png")
        lms_paths = _sorted_files(lms_dir, ".lms")

        n_available = min(len(img_paths), len(lms_paths))
        if n_available < 2:
            raise FileNotFoundError(f"Need at least 2 image/landmark pairs under {img_dir} and {lms_dir}.")

        # The legacy preprocessing often leaves the final landmark frame less reliable.
        n_frames = n_available - 1 if n_available > 2 and not include_last_frame else n_available
        if max_frames is not None:
            n_frames = min(n_frames, max(2, int(max_frames)))
        self.source_image_paths = img_paths[:n_frames]
        if lazy_assets:
            boxes = [_read_landmarks_to_bbox(str(path)) for path in lms_paths[:n_frames]]
            return ImageFrames(self.source_image_paths), boxes
        images: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []
        for img_path, lms_path in zip(img_paths[:n_frames], lms_paths[:n_frames]):
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            images.append(img)
            boxes.append(_read_landmarks_to_bbox(str(lms_path)))
        return images, boxes

    def _init_wenet_encoder(self, encoder_path: str):
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Wenet encoder ONNX not found: {encoder_path}")
        self.ort_ae = onnxruntime.InferenceSession(encoder_path, providers=self.providers)
        self._reset_wenet_cache()

    def _init_feather_encoder(self, encoder_onnx: str, checkpoint: str):
        if encoder_onnx:
            if not os.path.exists(encoder_onnx):
                raise FileNotFoundError(f"FeatherHuBERT ONNX not found: {encoder_onnx}")
            self.ort_ae = onnxruntime.InferenceSession(encoder_onnx, providers=self.providers)
            return

        if not checkpoint:
            raise ValueError("--feather_hubert_checkpoint is required for --asr feather_hubert without --encoder_onnx.")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"FeatherHuBERT checkpoint not found: {checkpoint}")

        import torch
        from data_utils.feather_hubert.feather_hubert import (
            expected_hubert_frames,
            get_best_device,
            load_feather_hubert,
            make_even_first_dim,
            normalize_waveform,
        )

        self._torch = torch
        self.feather_device = get_best_device()
        self.feather_model = load_feather_hubert(checkpoint, device=self.feather_device)
        self.feather_output_dim = int(self.feather_model.config.output_dim)
        if self.feather_output_dim != FEATHER_DEFAULT_OUTPUT_DIM:
            raise ValueError(
                f"FeatherHuBERT output_dim={self.feather_output_dim}, but the released UNet expects 1024."
            )
        self._feather_expected_frames = expected_hubert_frames
        self._feather_make_even = make_even_first_dim
        self._feather_normalize = normalize_waveform

    def _reset_wenet_cache(self):
        self.offset = np.ones((1,), dtype=np.int64) * 100
        self.att_cache = np.zeros([3, 8, 16, 128], dtype=np.float32)
        self.cnn_cache = np.zeros([3, 1, 512, 14], dtype=np.float32)

    def _reset_audio_state(self):
        if self.asr == ASR_WENET:
            self._reset_wenet_cache()
            self.audio_play_list: List[int] = [0] * PLAY_PRE_PAD
            self.audio_queue_get_feat = np.zeros([PRE_AUDIO_LEN], dtype=np.int16)
            self.using_feat = np.zeros([USING_FEAT_INIT, 16, 512], dtype=np.float32)
        else:
            self.audio_play_list = []
            self.audio_queue_get_feat = np.zeros([0], dtype=np.int16)
            self.using_feat = np.zeros([USING_FEAT_INIT, 2, self.feather_output_dim], dtype=np.float32)
            self.feather_audio_buffer = np.zeros([0], dtype=np.float32)
            self.feather_emitted_frames = 0
            self.feather_next_window_start = 0
            self.pending_feature_frames = 0

    def reset(self):
        self._reset_audio_state()
        self.counter = 0
        self.is_processing = True

    def reset_realtime_audio(self) -> None:
        """Start a new utterance without changing the shared playback cursor."""
        self._reset_audio_state()

    def _encode_feather_waveform(self, speech: np.ndarray) -> np.ndarray:
        """Encode one contextual waveform using the same model in either runtime."""
        if self.ort_ae is not None:
            name = self.ort_ae.get_inputs()[0].name
            return self.ort_ae.run(None, {name: speech[None, :]})[0][0].astype(np.float32)
        if self.feather_model is None or self._torch is None:
            raise RuntimeError("FeatherHuBERT checkpoint model is not initialized.")
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(speech).to(self.feather_device)[None]
            return self.feather_model(tensor)[0].detach().cpu().numpy().astype(np.float32)

    def prepare_offline_features(self, audio: np.ndarray) -> np.ndarray:
        """Use the offline extractor on a complete, unpadded PCM16 reply.

        FeatherHuBERT's GroupNorm uses temporal statistics, so short streaming
        windows cannot reproduce features extracted from a complete utterance.
        Playback may pad its final PCM packet; encoder input must not be padded.
        """
        from data_utils.feather_hubert.feather_hubert import (
            get_feather_hubert_from_16k_speech, make_even_first_dim,
        )

        if self.asr != ASR_FEATHER or self.feather_model is None:
            raise RuntimeError("Offline quality requires the FeatherHuBERT checkpoint model.")
        hop = FEATHER_STREAM_HOP_SAMPLES
        frame_count = (len(audio) + hop - 1) // hop
        hidden = get_feather_hubert_from_16k_speech(
            audio.astype(np.float32) / 32768.0,
            self.feather_model, device=self.feather_device,
        )
        features = make_even_first_dim(hidden).numpy().reshape(-1, 2, self.feather_output_dim)
        windows = np.zeros((frame_count, UNET_FEAT_WINDOW, 2, self.feather_output_dim), np.float32)
        for index in range(frame_count):
            left = index - USING_FEAT_INIT
            start, end = max(0, left), min(len(features), left + UNET_FEAT_WINDOW)
            if end > start:
                windows[index, start - left:end - left] = features[start:end]
        return windows

    def prepare_realtime_features(self, audio: np.ndarray, frame_count: int) -> np.ndarray:
        """Encode history + a playback batch + real lookahead in one model call.

        Only the playback batch enters history. Lookahead remains in the caller's
        audio queue and is encoded again with the next overlapping window.
        Windows match training: four past frames, the current frame, three future
        frames. At utterance boundaries missing features are zero, never repeated.
        """
        if self.asr != ASR_FEATHER:
            raise ValueError("Realtime packet inference requires FeatherHuBERT.")
        hop = FEATHER_STREAM_HOP_SAMPLES
        if frame_count <= 0 or len(audio) < frame_count * hop:
            raise ValueError("Realtime audio must contain the entire playback batch.")
        audio_float = audio.astype(np.float32) / 32768.0
        history_frames = len(self.feather_audio_buffer) // hop
        speech = np.concatenate([self.feather_audio_buffer, audio_float])
        real_frames = len(speech) // hop
        # 80 extra waveform samples produce both tokens for the final interval.
        required = real_frames * hop + (FEATHER_STREAM_WINDOW_SAMPLES - hop)
        if len(speech) < required:
            speech = np.pad(speech, (0, required - len(speech)))
        speech = (speech - speech.mean()) / np.sqrt(speech.var() + 1e-7)
        hidden = self._encode_feather_waveform(speech)
        if len(hidden) < real_frames * 2:
            raise RuntimeError("FeatherHuBERT returned too few tokens for the audio window.")
        features = hidden[:real_frames * 2].reshape(real_frames, 2, self.feather_output_dim)
        windows = np.zeros((frame_count, UNET_FEAT_WINDOW, 2, self.feather_output_dim), np.float32)
        for index in range(frame_count):
            left = history_frames + index - USING_FEAT_INIT
            start, end = max(0, left), min(real_frames, left + UNET_FEAT_WINDOW)
            windows[index, start - left:end - left] = features[start:end]
        self.feather_audio_buffer = np.concatenate([
            self.feather_audio_buffer, audio_float[:frame_count * hop]
        ])[-FEATHER_REALTIME_HISTORY_SAMPLES:]
        return windows

    def process_realtime(self, audio: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Render one frame with features prepared for its matching audio interval."""
        if features is None:
            features = self.prepare_realtime_features(audio, 1)[0]
        history = self.using_feat
        self.using_feat = features
        cached_input = getattr(self, "_realtime_image_input", None)
        try:
            self.last_frame_index = self.frame_picker.index
            self._realtime_image_input = self.prepare_frame_input(self.last_frame_index)
            return self._run_unet(
                self.full_body_img_list[self.last_frame_index].copy(),
                self.bbox_list[self.last_frame_index],
            )
        finally:
            self.using_feat = history
            self._realtime_image_input = cached_input

    def prepare_frame_input(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cache static FP32 inputs; audio features remain specific to each frame."""
        if not hasattr(self, "_realtime_input_cache"):
            self._realtime_input_cache = OrderedDict()
        cache = self._realtime_input_cache
        if index not in cache:
            cache[index] = _build_unet_inputs(crop_face(self.full_body_img_list[index], self.bbox_list[index]))
        cache.move_to_end(index)
        while len(cache) > max(1, getattr(self, "realtime_input_cache_size", 64)):
            expired, _ = cache.popitem(last=False)
            getattr(self, "_realtime_gpu_inputs", {}).pop(expired, None)
        return cache[index]

    def enable_realtime_io_binding(self) -> bool:
        """Reuse FP32 CUDA tensors when CUDA is available; retain normal CPU execution."""
        self.realtime_io_binding = "CUDAExecutionProvider" in self.ort_unet.get_providers()
        return self.realtime_io_binding

    def supports_realtime_batch(self) -> bool:
        shape = self.ort_unet.get_inputs()[0].shape
        return bool(shape and not isinstance(shape[0], int))

    def process_realtime_batch(self, frames: List[Tuple[int, np.ndarray]]) -> List[np.ndarray]:
        """Optional FP32 batching for exports with a dynamic batch dimension."""
        if not frames:
            return []
        cached = [self.prepare_frame_input(index) for index, _ in frames]
        image_input = np.concatenate([image for image, _ in cached], axis=0)
        audio_input = np.stack([features.reshape(16, 32, 32) for _, features in frames])
        output = self.ort_unet.run(None, {
            self.ort_unet.get_inputs()[0].name: image_input,
            self.ort_unet.get_inputs()[1].name: audio_input,
        })[0]
        images = []
        for (index, _), (_, original), prediction in zip(frames, cached, output):
            image = self.full_body_img_list[index].copy()
            xmin, ymin, xmax, ymax = self.bbox_list[index]
            crop_h, crop_w = image[ymin:ymax, xmin:xmax].shape[:2]
            crop = original.copy()
            crop[FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE, FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE] = (
                prediction.transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            image[ymin:ymax, xmin:xmax] = cv2.resize(crop, (crop_w, crop_h))
            images.append(image)
        return images

    def _run_bound_unet(self, image: np.ndarray, audio: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_realtime_gpu_inputs"):
            self._realtime_gpu_inputs = {}
            self._realtime_gpu_audio = onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(audio), "cuda", 0)
            self._realtime_gpu_output = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                [1, 3, FACE_INNER_SIZE, FACE_INNER_SIZE], np.float32, "cuda", 0,
            )
        index = self.last_frame_index
        if index not in self._realtime_gpu_inputs:
            self._realtime_gpu_inputs[index] = onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(image), "cuda", 0)
        self._realtime_gpu_audio.update_inplace(np.ascontiguousarray(audio))
        binding = self.ort_unet.io_binding()
        binding.bind_ortvalue_input(self.ort_unet.get_inputs()[0].name, self._realtime_gpu_inputs[index])
        binding.bind_ortvalue_input(self.ort_unet.get_inputs()[1].name, self._realtime_gpu_audio)
        binding.bind_ortvalue_output(self.ort_unet.get_outputs()[0].name, self._realtime_gpu_output)
        self.ort_unet.run_with_iobinding(binding)
        return self._realtime_gpu_output.numpy()

    def _detect_silence(self, audio_frame: np.ndarray):
        if not np.any(audio_frame):
            if not self.silence:
                self.empty_audio_counter += 1
            if self.empty_audio_counter >= SILENCE_THRESHOLD:
                self.silence = True
        else:
            self.empty_audio_counter = 0
            self.silence = False

    def _next_idle_img(self) -> Tuple[Optional[np.ndarray], int]:
        if self.counter == 0:
            self.last_frame_index = self.frame_picker.index
            img = self.full_body_img_list[self.last_frame_index].copy()
            self.frame_picker.advance()
            self.counter = 1
            return img, 1
        self.counter += 1
        if self.counter == self.idle_loop:
            self.counter = 0
        return None, 0

    def _next_silent_idle_img(self) -> Tuple[Optional[np.ndarray], int]:
        return self._next_idle_img()

    def warm_up(self, chunks: int = 80) -> None:
        picker_index = self.frame_picker.index
        picker_step = self.frame_picker.step
        for i in range(max(1, chunks)):
            t = np.arange(FRAME_LEN) + i * FRAME_LEN
            audio_frame = (np.sin(2 * np.pi * 440 * t / SAMPLE_RATE) * 12000).astype(np.int16)
            self.process(audio_frame)

        self._reset_audio_state()
        self.counter = 0
        self.empty_audio_counter = SILENCE_THRESHOLD
        self.is_processing = False
        self.silence = True
        self.last_frame_index = picker_index
        self.frame_picker.index = picker_index
        self.frame_picker.step = picker_step

    def _pop_play_audio(self) -> np.ndarray:
        if self.audio_play_list:
            audio = np.array(self.audio_play_list[:FRAME_LEN], dtype=np.int16)
            self.audio_play_list = self.audio_play_list[FRAME_LEN:]
            if audio.shape[0] < FRAME_LEN:
                audio = np.pad(audio, (0, FRAME_LEN - audio.shape[0]))
            return audio
        return np.zeros([FRAME_LEN], dtype=np.int16)

    def _run_wenet_encoder(self) -> np.ndarray:
        if self.ort_ae is None:
            raise RuntimeError("Wenet encoder is not initialized.")
        fbank = knf.OnlineFbank(_get_fbank_opts())
        fbank.accept_waveform(SAMPLE_RATE, self.audio_queue_get_feat.tolist())
        self.audio_play_list.extend(
            self.audio_queue_get_feat[PRE_AUDIO_LEN:PRE_AUDIO_LEN + WENET_CHUNK_DROP].tolist()
        )

        mel = np.array([[fbank.get_frame(i) for i in range(fbank.num_frames_ready)]])
        mel = mel[:, :, :WENET_FEAT_FRAMES, :]
        inputs = {
            "chunk": mel.astype(np.float32),
            "offset": self.offset,
            "att_cache": self.att_cache.astype(np.float32),
            "cnn_cache": self.cnn_cache.astype(np.float32),
        }
        outs = self.ort_ae.run(None, inputs)
        return outs[0].astype(np.float32)

    def _run_feather_encoder(self, pad_right_context: bool = False) -> np.ndarray:
        if self._feather_expected_frames is None:
            from data_utils.feather_hubert.feather_hubert import expected_hubert_frames

            self._feather_expected_frames = expected_hubert_frames

        if self.ort_ae is not None:
            features = []
            input_name = self.ort_ae.get_inputs()[0].name
            normalized_buffer = self.feather_audio_buffer.astype(np.float32)
            if normalized_buffer.shape[0] > 0:
                normalized_buffer = (
                    normalized_buffer - normalized_buffer.mean()
                ) / np.sqrt(normalized_buffer.var() + 1e-7)
                if pad_right_context:
                    # The 720-sample window needs 80 samples beyond the current
                    # 640-sample video interval. Pad only this encoder input;
                    # retaining padding in waveform history would shift speech.
                    normalized_buffer = np.pad(
                        normalized_buffer,
                        (0, FEATHER_STREAM_WINDOW_SAMPLES - FEATHER_STREAM_HOP_SAMPLES),
                        mode="edge",
                    )
            while normalized_buffer.shape[0] >= self.feather_next_window_start + FEATHER_STREAM_WINDOW_SAMPLES:
                start = self.feather_next_window_start
                end = start + FEATHER_STREAM_WINDOW_SAMPLES
                speech = normalized_buffer[start:end]
                hidden = self.ort_ae.run(None, {input_name: speech[None, :]})[0][0].astype(np.float32)
                if hidden.shape[0] >= FEATHER_TOKENS_PER_VIDEO_FRAME:
                    features.append(hidden[:FEATHER_TOKENS_PER_VIDEO_FRAME])
                self.feather_next_window_start += FEATHER_STREAM_HOP_SAMPLES

            if self.feather_next_window_start >= SAMPLE_RATE:
                trim = self.feather_next_window_start
                self.feather_audio_buffer = self.feather_audio_buffer[trim:]
                self.feather_next_window_start = 0

            if not features:
                return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)
            return np.stack(features, axis=0).astype(np.float32)

        expected_tokens = self._feather_expected_frames(int(self.feather_audio_buffer.shape[0]))
        expected_video_frames = expected_tokens // FEATHER_TOKENS_PER_VIDEO_FRAME
        if expected_video_frames <= self.feather_emitted_frames:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        if self.feather_model is None or self._torch is None:
            raise RuntimeError("FeatherHuBERT checkpoint model is not initialized.")
        speech = self._feather_normalize(self.feather_audio_buffer)
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(speech).to(self.feather_device)[None]
            hidden_tensor = self.feather_model(tensor)[0].detach().cpu()
            hidden = self._feather_make_even(hidden_tensor).numpy().astype(np.float32)

        if hidden.shape[0] < FEATHER_TOKENS_PER_VIDEO_FRAME:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        video_features = hidden.reshape(-1, FEATHER_TOKENS_PER_VIDEO_FRAME, hidden.shape[-1])
        stable_frames = max(0, video_features.shape[0] - self.feather_right_context_frames)
        if stable_frames <= self.feather_emitted_frames:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        new_features = video_features[self.feather_emitted_frames:stable_frames]
        self.feather_emitted_frames = stable_frames
        return new_features.astype(np.float32)

    def _audio_feat_for_unet(self) -> np.ndarray:
        if self.asr == ASR_WENET:
            return self.using_feat.reshape(1, 128, 16, 32)
        return self.using_feat.reshape(1, 16, 32, 32)

    def _run_idle_unet(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        old_using_feat = self.using_feat
        if self.asr == ASR_WENET:
            self.using_feat = np.zeros([UNET_FEAT_WINDOW, 16, 512], dtype=np.float32)
        else:
            self.using_feat = np.zeros([UNET_FEAT_WINDOW, 2, self.feather_output_dim], dtype=np.float32)
        try:
            return self._run_unet(img, bbox)
        finally:
            self.using_feat = old_using_feat

    def _run_unet(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        xmin, ymin, xmax, ymax = bbox
        crop_h, crop_w = img[ymin:ymax, xmin:xmax].shape[:2]
        cached = getattr(self, "_realtime_image_input", None)
        if cached is None:
            onnx_in, crop_ori = _build_unet_inputs(crop_face(img, bbox))
        else:
            onnx_in, original = cached
            crop_ori = original.copy()

        audio_feat = self._audio_feat_for_unet()
        inputs = {
            self.ort_unet.get_inputs()[0].name: onnx_in,
            self.ort_unet.get_inputs()[1].name: audio_feat,
        }
        if cached is not None and getattr(self, "realtime_io_binding", False):
            try:
                output = self._run_bound_unet(onnx_in, audio_feat)
            except Exception as exc:
                # Some exported graphs/runtimes cannot bind CUDA tensors. Keep
                # the exact FP32 model and continue through the ordinary session.
                self.realtime_io_binding = False
                self._realtime_gpu_inputs = {}
                warnings.warn(f"CUDA I/O binding disabled: {exc}", RuntimeWarning)
                output = self.ort_unet.run(None, inputs)[0]
        else:
            output = self.ort_unet.run(None, inputs)[0]
        pred = (output[0].transpose(1, 2, 0) * 255).astype(np.uint8)

        b = FACE_BORDER
        crop_ori[b:b + FACE_INNER_SIZE, b:b + FACE_INNER_SIZE] = pred
        crop_ori = cv2.resize(crop_ori, (crop_w, crop_h))
        img[ymin:ymax, xmin:xmax] = crop_ori
        return img

    def _process_wenet(self, audio_frame: np.ndarray):
        self.audio_queue_get_feat = np.concatenate([self.audio_queue_get_feat, audio_frame], axis=0)
        if self.audio_queue_get_feat.shape[0] >= WENET_TRIGGER_LEN:
            audio_feat = self._run_wenet_encoder()
            self.audio_queue_get_feat = self.audio_queue_get_feat[WENET_CHUNK_DROP:]
            self.using_feat = np.concatenate([self.using_feat, audio_feat], axis=0)

            img = self.full_body_img_list[self.frame_picker.index].copy()
            bbox = self.bbox_list[self.frame_picker.index]
            self.frame_picker.advance()
            if self.using_feat.shape[0] >= UNET_FEAT_WINDOW:
                img = self._run_unet(img, bbox)
                self.using_feat = self.using_feat[1:]
            self.counter = 1
            return img.copy(), self._pop_play_audio(), 1

        return_img, check_img = self._next_idle_img()
        return return_img, self._pop_play_audio(), check_img

    def _process_feather(self, audio_frame: np.ndarray):
        self.audio_play_list.extend(audio_frame.tolist())
        audio_float = audio_frame.astype(np.float32) / 32768.0
        self.feather_audio_buffer = np.concatenate([self.feather_audio_buffer, audio_float], axis=0)

        new_features = self._run_feather_encoder()
        if new_features.shape[0] > 0:
            self.using_feat = np.concatenate([self.using_feat, new_features], axis=0)
            self.pending_feature_frames += int(new_features.shape[0])

        if self.pending_feature_frames > 0 and self.using_feat.shape[0] >= UNET_FEAT_WINDOW:
            img = self.full_body_img_list[self.frame_picker.index].copy()
            bbox = self.bbox_list[self.frame_picker.index]
            self.frame_picker.advance()
            img = self._run_unet(img, bbox)
            self.using_feat = self.using_feat[1:]
            self.pending_feature_frames -= 1
            self.counter = 1
            return img.copy(), self._pop_play_audio(), 1

        return_img, check_img = self._next_idle_img()
        return return_img, self._pop_play_audio(), check_img

    def process(self, audio_frame: np.ndarray):
        audio_frame = audio_frame.astype(np.int16)
        if audio_frame.shape[0] < FRAME_LEN:
            audio_frame = np.pad(audio_frame, (0, FRAME_LEN - audio_frame.shape[0]))
        elif audio_frame.shape[0] > FRAME_LEN:
            audio_frame = audio_frame[:FRAME_LEN]

        self._detect_silence(audio_frame)
        if self.silence:
            if self.is_processing:
                self._reset_audio_state()
            self.is_processing = False
            return_img, check_img = self._next_silent_idle_img()
            return return_img, np.zeros([FRAME_LEN], dtype=np.int16), check_img

        if not self.is_processing:
            self.reset()

        if self.asr == ASR_WENET:
            return self._process_wenet(audio_frame)
        return self._process_feather(audio_frame)


def _select_fourcc(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc("M", "J", "P", "G")
    return cv2.VideoWriter_fourcc(*"mp4v")


def _resize_for_writer(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h))


def main(arg):
    import soundfile as sf
    from scipy.io import wavfile

    asr = _normalize_asr(arg.asr)
    stream, sr = sf.read(arg.audio_wav)
    if stream.ndim == 2:
        stream = stream[:, 0]
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz wav, got {sr} Hz: {arg.audio_wav}")
    stream = (stream.astype(np.float32) * 32767).astype(np.int16)

    w, h = arg.video_size
    fps = arg.fps if arg.fps > 0 else FPS_BY_ASR[asr]
    writer = cv2.VideoWriter(arg.out_video, _select_fourcc(arg.out_video), fps, (w, h))
    processor = DiHumanProcessor(
        arg.data_path,
        asr=asr,
        unet_onnx=arg.unet_onnx,
        encoder_onnx=arg.encoder_onnx,
        feather_hubert_checkpoint=arg.feather_hubert_checkpoint,
        feather_right_context_frames=arg.feather_right_context_frames,
        force_cpu_onnx=arg.cpu_onnx,
    )

    audio_out: List[np.ndarray] = []
    n_chunks = math.ceil(stream.shape[0] / FRAME_LEN)
    for i in range(n_chunks):
        a = i * FRAME_LEN
        b = min(a + FRAME_LEN, stream.shape[0])
        audio_frame = stream[a:b]
        img, playing_audio, check_img = processor.process(audio_frame)
        audio_out.append(playing_audio)
        if check_img and img is not None:
            writer.write(_resize_for_writer(img, (w, h)))

    writer.release()
    audio_data = np.concatenate(audio_out).astype(np.int16)
    wavfile.write(arg.out_audio, SAMPLE_RATE, audio_data)
    print(f"[done] asr={asr}, fps={fps}, video saved to {arg.out_video}, audio saved to {arg.out_audio}")


if __name__ == "__main__":
    main(parse_args())


# Merge audio and video:
# ffmpeg -i test_video.mp4 -i test_audio.wav -c:v libx264 -c:a aac result_test.mp4

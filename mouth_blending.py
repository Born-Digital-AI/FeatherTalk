"""Inference-only mouth compositing using the training dataset's ROI geometry."""
from __future__ import annotations

import cv2
import numpy as np

from dataset_mouth_roi import MouthRoiConfig, mouth_roi_from_landmarks
from face_utils import FACE_BORDER, FACE_INNER_SIZE


DEFAULT_MOUTH_ROI = MouthRoiConfig(expand_x=1.5, expand_y=2.0)
DEFAULT_MOUTH_FEATHER = 10.0


def mouth_alpha_from_landmarks(
    landmarks: np.ndarray,
    config: MouthRoiConfig = DEFAULT_MOUTH_ROI,
    feather: float = DEFAULT_MOUTH_FEATHER,
) -> np.ndarray:
    """Rounded mouth ROI with a smooth, inward-only feather in inner-crop pixels.

    Distance to a rounded rectangle gives a flat central mouth region. Smoothstep
    reaches exactly zero at the boundary, without blur leaking into nearby skin.
    The ROI bounds and landmark projection come directly from the training code.
    """
    points = np.asarray(landmarks)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < max(53, config.end):
        raise ValueError("Mouth blending requires the complete face and mouth landmarks.")
    if not np.isfinite(points).all() or points[31, 0] <= points[1, 0]:
        raise ValueError("Mouth blending requires finite landmarks and a positive face width.")
    if not (0 <= config.start < config.end) or not all(
        np.isfinite(value) and value > 0 for value in (config.expand_x, config.expand_y, feather)
    ):
        raise ValueError("Mouth expansion and feather must be finite and positive.")
    x1, y1, x2, y2 = mouth_roi_from_landmarks(points, config)
    half_w, half_h = (x2 - x1) / 2, (y2 - y1) / 2
    radius = min(half_w, half_h, max(feather * 1.5, min(half_w, half_h) * 0.55))
    y, x = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    qx = np.abs(x + 0.5 - (x1 + x2) / 2) - (half_w - radius)
    qy = np.abs(y + 0.5 - (y1 + y2) / 2) - (half_h - radius)
    distance = radius - (
        np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        + np.minimum(np.maximum(qx, qy), 0)
    )
    # Keep a flat core even when clipping leaves a very narrow visible ROI.
    feather_width = min(feather, max(0.5, min(half_w, half_h) * 0.75))
    t = np.clip(distance / feather_width, 0, 1)
    alpha = np.zeros((FACE_INNER_SIZE, FACE_INNER_SIZE), np.float32)
    alpha[y1:y2, x1:x2] = t * t * (3 - 2 * t)
    return alpha


def blend_with_alpha(source: np.ndarray, prediction: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend RGB/BGR uint8 images, copying every zero-alpha pixel exactly."""
    if source.shape != prediction.shape or alpha.shape != source.shape[:2]:
        raise ValueError("Source, prediction and mouth alpha must have matching dimensions.")
    output = source.copy()
    active = alpha > 0
    weight = alpha[active, None]
    values = source[active].astype(np.float32) * (1 - weight) + prediction[active].astype(np.float32) * weight
    output[active] = np.clip(np.rint(values), 0, 255).astype(source.dtype)
    return output


def clip_face_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Visible input crop bounds, shared by preprocessing and prediction resize."""
    x1, y1, x2, y2 = bbox
    clipped = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Face bounding box must intersect the source image.")
    return clipped


def paste_predicted_mouth(
    image: np.ndarray,
    prediction: np.ndarray,
    face_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: np.ndarray,
    config: MouthRoiConfig = DEFAULT_MOUTH_ROI,
    feather: float = DEFAULT_MOUTH_FEATHER,
) -> np.ndarray:
    """Composite onto native source pixels; return the full-frame alpha mask.

    Resizing an already blended face would change skin outside the mouth through
    a downscale/upscale round trip. Resize prediction and alpha separately, then
    blend with the original frame. Only alpha-positive pixels can change.
    """
    inner_alpha = mouth_alpha_from_landmarks(landmarks, config, feather)
    alpha = np.pad(inner_alpha, FACE_BORDER)
    generated = face_crop.copy()
    b = FACE_BORDER
    generated[b:b + FACE_INNER_SIZE, b:b + FACE_INNER_SIZE] = prediction
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Mouth paste-back requires a positive face bounding box.")
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    # Safely intersect with the image if a face box crosses the frame boundary.
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(image.shape[1], x2), min(image.shape[0], y2)
    full_alpha = np.zeros(image.shape[:2], np.float32)
    if right <= left or bottom <= top:
        return full_alpha
    alpha = alpha[top-y1:bottom-y1, left-x1:right-x1]
    # crop_face resizes the visible source region, not the nominal out-of-frame
    # box. Invert that same transform for predictions. Alpha uses the nominal
    # landmark transform so its native support still follows the visible mouth.
    generated = cv2.resize(generated, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
    original = image[top:bottom, left:right]
    image[top:bottom, left:right] = blend_with_alpha(original, generated, alpha)
    full_alpha[top:bottom, left:right] = alpha
    return full_alpha


def add_mouth_cli_arguments(parser) -> None:
    """Shared inference flags; model inputs and training are unaffected."""
    def positive_float(value):
        import argparse

        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError("must be finite and greater than zero")
        return number

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--mouth-only-pasteback', dest='mouth_only_pasteback', action='store_true',
                       help='Blend only the landmark mouth ROI into the original frame (default).')
    modes.add_argument('--full-face-pasteback', dest='mouth_only_pasteback', action='store_false',
                       help='Use legacy replacement of the entire inner face for comparison.')
    parser.set_defaults(mouth_only_pasteback=True)
    parser.add_argument('--mouth-expand-x', type=positive_float, default=DEFAULT_MOUTH_ROI.expand_x)
    parser.add_argument('--mouth-expand-y', type=positive_float, default=DEFAULT_MOUTH_ROI.expand_y)
    parser.add_argument('--mouth-feather', type=positive_float, default=DEFAULT_MOUTH_FEATHER,
                        help='Inward feather width in 160x160 inner-crop pixels.')


def save_mouth_debug(directory, frame_index: int, source: np.ndarray, alpha: np.ndarray) -> None:
    """Save lossless full-resolution alpha and source-frame ROI overlay PNGs."""
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    overlay = blend_with_alpha(source, np.full_like(source, (0, 220, 0)), alpha * 0.5)
    mask = np.rint(alpha * 255).astype(np.uint8)
    for suffix, frame in [('alpha', mask), ('overlay', overlay)]:
        path = root / f'frame_{frame_index:05d}_{suffix}.png'
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Cannot write mouth debug image: {path}")

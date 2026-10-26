"""Image preprocessing pipeline shared by training, feature extraction and inference.

read -> validate -> RGB -> eye ROI -> resize 224x224 -> illumination
normalisation -> CLAHE -> (ImageNet normalisation + tensor, done in to_tensor).

Two outputs are produced:
  roi_raw   : 224x224 RGB after cropping/resizing only. Handcrafted features are
              measured here so that CLAHE does not inflate sharpness/contrast.
  enhanced  : roi_raw after illumination normalisation + CLAHE. Fed to ResNet-18.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from preprocessing import eye_roi


@dataclass
class PreprocessResult:
    roi_raw: np.ndarray
    enhanced: np.ndarray
    eye_found: bool = False
    num_eyes: int = 0
    roi_box: Optional[Tuple[int, int, int, int]] = None
    notes: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ I/O
def read_image_rgb(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read an image file as RGB uint8. Returns None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with unicode paths on Windows
        return decode_image_bytes(data.tobytes())
    except Exception:
        return None


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # always 3-channel; handles PNG alpha / grayscale
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------- operations
def resize_square(rgb: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def normalize_illumination(rgb: np.ndarray, blend: float = config.ILLUMINATION_BLEND) -> np.ndarray:
    """Gentle flat-fielding of the lightness channel (removes slow shading gradients)."""
    if blend <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    sigma = 0.25 * min(L.shape)
    background = cv2.GaussianBlur(L, (0, 0), sigma)
    flat = np.clip(L / (background + 1e-6) * L.mean(), 0, 255)
    lab[..., 0] = np.clip((1 - blend) * L + blend * flat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_clahe(rgb: np.ndarray, clip: float = config.CLAHE_CLIP_LIMIT,
                tiles: Tuple[int, int] = config.CLAHE_TILE_GRID) -> np.ndarray:
    """CLAHE on the L channel only, so iris colour (a/b channels) is untouched."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_imagenet(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 HxWx3 normalised with ImageNet mean/std."""
    x = rgb.astype(np.float32) / 255.0
    return (x - np.array(config.IMAGENET_MEAN, np.float32)) / np.array(config.IMAGENET_STD, np.float32)


def to_tensor(rgb: np.ndarray):
    """uint8 HxWx3 -> normalised float tensor 3xHxW (deterministic path used at inference)."""
    import torch
    return torch.from_numpy(normalize_imagenet(rgb).transpose(2, 0, 1).copy())


# ------------------------------------------------------------ pipeline
def preprocess(rgb: np.ndarray, mode: str = "auto") -> PreprocessResult:
    """Run the full pipeline.

    mode="dataset": images are already tight eye crops -> centre-square + resize only.
    mode="auto"   : user photos -> crop wider photos to the eye; preserve framing if already a close-up.
    """
    notes: List[str] = []
    found, num_eyes, box = False, 0, None
    h, w = rgb.shape[:2]

    if mode == "auto":
        boxes = eye_roi.detect_eyes(rgb)
        num_eyes = eye_roi.count_similar_eyes(boxes)

        # An image is already a close-up eye photo if resolution is <= 400px
        # or if the detected ocular feature occupies a substantial fraction of the image.
        is_already_close_up = (min(h, w) <= 400) or (
            bool(boxes) and (boxes[0][2] >= 0.35 * w or boxes[0][2] * boxes[0][3] >= 0.15 * h * w)
        )

        if boxes and not is_already_close_up:
            found, box = True, boxes[0]
            crop = eye_roi.crop_around_eye(rgb, box, config.EYE_FRAME_FILL)
            notes.append("Ocular region isolated and centered.")
        else:
            found = bool(boxes) or (min(h, w) <= 400)
            box = boxes[0] if boxes else None
            notes.append("Close-up eye framing preserved." if is_already_close_up else "Center framing used.")
            crop = eye_roi.center_square(rgb)
    else:
        crop = eye_roi.center_square(rgb)

    roi_raw = resize_square(crop)
    enhanced = apply_clahe(normalize_illumination(roi_raw))
    return PreprocessResult(roi_raw, enhanced, found, num_eyes, box, notes)

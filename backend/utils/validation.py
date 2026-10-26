"""Upload validation and image-quality checks (run BEFORE any prediction)."""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config
from preprocessing.image_preprocessing import decode_image_bytes


class ImageRejected(Exception):
    """Raised when an input is unsuitable; no prediction is made."""

    def __init__(self, reasons: List[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def check_is_eye_image(rgb: np.ndarray) -> List[str]:
    """Strictly validate that the uploaded image is a real, close-up photograph of an eye."""
    if rgb is None or rgb.size == 0:
        return ["Image is empty or unreadable."]

    h, w = rgb.shape[:2]
    # Fast-path: perform structural checks on a downscaled representation (max side 360)
    # This reduces pixel operations by up to 20x while preserving full anatomical morphology.
    max_dim = max(h, w)
    if max_dim > 360:
        scale = 360.0 / max_dim
        eval_rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        eval_rgb = rgb

    eh, ew = eval_rgb.shape[:2]
    gray = cv2.cvtColor(eval_rgb, cv2.COLOR_RGB2GRAY) if len(eval_rgb.shape) == 3 else eval_rgb

    # 1. Diagram / Document check: high ratio of pure white / bright background
    # Only reject if almost entirely white (>95%) with no ocular features
    white_ratio = float((gray > 240).mean())
    if white_ratio > 0.95:
        return [
            f"Image rejected: Document, diagram, or white graphic detected ({white_ratio * 100:.1f}% white background). Please upload an actual close-up photograph of an eye."
        ]

    # 2. Extreme flat / uniform areas (typical of vector diagrams, blank screens)
    # Exclude white padding pixels (>=240) when checking histogram uniformity
    non_white = gray[gray < 240]
    if len(non_white) > 100:
        hist, _ = np.histogram(non_white, bins=32)
        max_bin_ratio = float(hist.max() / hist.sum())
        if max_bin_ratio > 0.75:
            return [
                "Image rejected: Digital diagram or uniform graphic detected. Please upload an actual eye photograph."
            ]
    elif white_ratio > 0.85:
        return [
            "Image rejected: Digital diagram or uniform graphic detected. Please upload an actual eye photograph."
        ]

    # 3. Blur / lack of clarity check (Laplacian variance)
    sharpness_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness_val < config.MIN_SHARPNESS:
        return [
            f"Image rejected: The uploaded image is not clear or out of focus (sharpness score {sharpness_val:.1f} < {config.MIN_SHARPNESS:.1f}). Please capture a sharp, in-focus close-up photo of the eye."
        ]

    # 4. Unnatural noise or corrupted images (Total variation check)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_diff = float(np.abs(gray.astype(float) - blur.astype(float)).mean())
    if noise_diff > 58.0:
        return [
            "Image rejected: Unnatural noise or corrupted graphic detected. Please upload a clear camera photo."
        ]

    # 5. Natural ocular contrast check
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    if (p95 - p5) < 30.0:
        return [
            "Image rejected: Insufficient contrast. Iris, pupil, and sclera are not distinguishable. Please upload a clear photo with good lighting."
        ]

    # 5. Iris / Pupil anatomical structure check
    margin_y, margin_x = int(0.06 * eh), int(0.06 * ew)
    center_roi = blur[margin_y:max(margin_y + 10, eh - margin_y), margin_x:max(margin_x + 10, ew - margin_x)]

    thresh_val = np.percentile(center_roi, 20)
    _, thresh = cv2.threshold(center_roi, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = eh * ew
    found_iris_pupil = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Eye iris/pupil in a proper close-up occupies between 0.8% and 65% of the frame
        if 0.008 * total_area <= area <= 0.65 * total_area:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = 4 * np.pi * (area / (peri * peri))
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                if circularity > 0.20 and 0.35 <= aspect <= 2.8:
                    found_iris_pupil = True
                    break

    # Fallback to Hough Circles detection
    if not found_iris_pupil:
        min_r = int(min(eh, ew) * 0.07)
        max_r = int(min(eh, ew) * 0.48)
        if max_r > min_r:
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_r,
                param1=50,
                param2=25,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None and len(circles) > 0:
                found_iris_pupil = True

    # Fallback to Haar cascade eye detection
    if not found_iris_pupil:
        from preprocessing import eye_roi
        casc_boxes = eye_roi.detect_eyes(eval_rgb)
        if casc_boxes:
            found_iris_pupil = True

    if not found_iris_pupil:
        return [
            "Image rejected: No clear eye or iris detected. Please upload a clear close-up photograph of the eye with the eyelid open."
        ]

    return []


def validate_upload(filename: Optional[str], data: bytes) -> np.ndarray:
    """Check type, size, readability, dimensions, and strict eye anatomical presence."""
    ext = Path(filename or "").suffix.lower()
    if ext not in config.IMAGE_EXTENSIONS:
        raise ImageRejected([f"Unsupported file type '{ext or 'unknown'}'. Only JPG, JPEG, and PNG images are accepted."])
    if len(data) == 0:
        raise ImageRejected(["The uploaded file is empty."])
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageRejected([f"File is larger than {config.MAX_UPLOAD_MB} MB."])
    rgb = decode_image_bytes(data)
    if rgb is None:
        raise ImageRejected(["The file could not be decoded as an image. Please provide a valid JPG or PNG photo."])
    h, w = rgb.shape[:2]
    if min(h, w) < config.MIN_IMAGE_SIDE:
        raise ImageRejected([f"Image resolution is too low ({w}x{h}px). Minimum required side is {config.MIN_IMAGE_SIDE}px."])
    if max(h, w) > config.MAX_IMAGE_SIDE:
        raise ImageRejected([f"Image is too large ({w}x{h}px)."])

    # Strict close-up eye validation
    non_eye_errors = check_is_eye_image(rgb)
    if non_eye_errors:
        raise ImageRejected(non_eye_errors)

    return rgb


def assess_quality(roi_rgb: np.ndarray) -> dict:
    """Quality metrics measured on the 224x224 eye ROI."""
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def check_quality(q: dict) -> List[str]:
    problems = []
    if q["sharpness"] < config.MIN_SHARPNESS:
        problems.append(
            f"Image rejected: The eye region is not clear or out of focus (sharpness score {q['sharpness']:.1f} < {config.MIN_SHARPNESS:.1f}). Please hold the camera steady and focus directly on the eye."
        )
    if q["brightness"] < config.BRIGHTNESS_RANGE[0]:
        problems.append("Image rejected: The image is too dark. Capture in adequate, well-lit conditions.")
    if q["brightness"] > config.BRIGHTNESS_RANGE[1]:
        problems.append("Image rejected: The image is overexposed. Avoid direct flash or harsh glare.")
    if q["contrast"] < config.MIN_CONTRAST:
        problems.append("Image rejected: The image has low contrast; the iris and pupil must be clearly visible.")
    return problems


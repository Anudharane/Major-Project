"""Eye region-of-interest (ROI) detection.

Uses OpenCV's bundled Haar eye cascade. It is only a helper:
  * on wide photos (face / half-face selfies) it lets us crop to the eye,
  * it is one piece of evidence that the image really contains an eye.
The dataset images are already tight eye crops, so training uses
mode="dataset" (no detection) to avoid inconsistent crops.
"""
from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h
_cascade = None
_MAX_DETECT_SIDE = 640


def _get_cascade():
    global _cascade
    if _cascade is None:
        try:
            cls = getattr(cv2, "CascadeClassifier", None)
            if cls is not None:
                # 1. Try OpenCV data
                data = getattr(cv2, "data", None)
                if data and hasattr(data, "haarcascades"):
                    path = data.haarcascades + "haarcascade_eye.xml"
                    c = cls(path)
                    if not c.empty():
                        _cascade = c
                        return _cascade
                # 2. Try project bundled cascades
                from pathlib import Path
                local_path = Path(__file__).resolve().parent.parent / "data" / "cascades" / "haarcascade_eye.xml"
                if local_path.exists():
                    c = cls(str(local_path))
                    if not c.empty():
                        _cascade = c
                        return _cascade
            _cascade = False
        except BaseException:
            _cascade = False
    return _cascade if (_cascade is not False and _cascade is not None) else None


def detect_eyes(rgb: np.ndarray) -> List[Box]:
    """Return eye boxes (in original image coordinates), largest first.
    
    Combines Haar cascades (when available) with dark iris/pupil contour and
    Hough circle detection to locate the eye accurately in close-up or selfie photos.
    """
    h, w = rgb.shape[:2]
    boxes: List[Box] = []

    # 1. Try Haar cascade if available
    casc = _get_cascade()
    if casc is not None:
        scale = min(1.0, _MAX_DETECT_SIDE / max(h, w))
        img = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        min_side = max(24, int(0.12 * min(gray.shape)))
        try:
            found = casc.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side)
            )
            for b in found:
                boxes.append(tuple(int(v / scale) for v in b))
        except Exception:
            pass

    # 2. Iris/pupil contour & circular feature detection (works on tight eye crops & macro photos)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh_val = np.percentile(blur, 18)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.008 * total_area <= area <= 0.60 * total_area:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = 4 * np.pi * (area / (peri * peri))
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                if circularity > 0.20 and 0.35 <= aspect <= 2.8:
                    # Pad slightly to capture the entire eye ROI
                    pad_w = int(bw * 0.35)
                    pad_h = int(bh * 0.35)
                    bx = max(0, x - pad_w)
                    by = max(0, y - pad_h)
                    bw_full = min(w - bx, bw + 2 * pad_w)
                    bh_full = min(h - by, bh + 2 * pad_h)
                    boxes.append((bx, by, bw_full, bh_full))

    # 3. Hough Circles fallback
    if not boxes:
        min_r = int(min(h, w) * 0.07)
        max_r = int(min(h, w) * 0.48)
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
                for c in circles[0]:
                    cx, cy, r = c
                    bx = int(max(0, cx - r * 1.4))
                    by = int(max(0, cy - r * 1.4))
                    side = int(min(w - bx, h - by, 2.8 * r))
                    if side > 16:
                        boxes.append((bx, by, side, side))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def count_similar_eyes(boxes: List[Box], ratio: float = 0.6) -> int:
    """Number of boxes whose area is within `ratio` of the largest and that do not overlap it."""
    if not boxes:
        return 0
    x0, y0, w0, h0 = boxes[0]
    n = 1
    for x, y, w, h in boxes[1:]:
        similar = (w * h) >= ratio * (w0 * h0)
        overlap = not (x + w <= x0 or x0 + w0 <= x or y + h <= y0 or y0 + h0 <= y)
        if similar and not overlap:
            n += 1
    return n


def crop_around_eye(rgb: np.ndarray, box: Box, fill: float) -> np.ndarray:
    """Square crop centred on the eye so that eye width ~= fill * crop width."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = int(round(max(bw, bh) / fill))
    side = max(8, min(side, min(h, w)))
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return rgb[y0:y0 + side, x0:x0 + side]


def center_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]


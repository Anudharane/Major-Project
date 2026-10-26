"""Synthetic test images (no dataset needed)."""
import cv2
import numpy as np


def synthetic_eye(size=224, pupil=18, iris=48, seed=0, blur=0, brightness=1.0, noise=6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (205, 160, 135), np.uint8)               # skin
    c = size // 2
    cv2.ellipse(img, (c, c), (int(size * .42), int(size * .24)), 0, 0, 360, (240, 238, 235), -1)  # sclera
    cv2.circle(img, (c, c), iris, (110, 85, 55), -1)                         # iris
    for a in range(0, 360, 12):                                              # iris texture
        cv2.line(img, (c, c), (int(c + iris * np.cos(np.radians(a))), int(c + iris * np.sin(np.radians(a)))), (140, 110, 75), 1)
    cv2.circle(img, (c, c), pupil, (8, 8, 8), -1)                            # pupil
    cv2.circle(img, (c - 14, c - 14), 5, (255, 255, 255), -1)                # specular highlight
    img = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def encode(rgb: np.ndarray, ext=".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()

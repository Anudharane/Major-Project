"""System test against a RUNNING backend (uvicorn main:app).

    python tests/system_test.py                       # synthetic bad-input checks
    python tests/system_test.py --folder my_images    # + your own images, see below

Optional folder layout (any subset): my_images/{normal,myopia,hyperopia,non_eye,blurry,dark}/*.jpg
Normal/myopia/hyperopia images are compared with the folder label (informational: labels are unverified);
non_eye / blurry / dark images are expected to be REJECTED (HTTP 422) or reported "Uncertain".
Camera capture is tested manually in the browser (see README).
"""
import argparse
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.helpers import encode, synthetic_eye


def post(client, name, data):
    return client.post("/predict", files={"file": (name, data)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--folder", type=Path)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, timeout=60)
    h = c.get("/health").json()
    print("health:", h)
    if not h.get("model_loaded"):
        sys.exit("Model not loaded - train first.")

    rng = np.random.default_rng(0)
    cases = {
        "invalid file (text)": ("a.txt", b"hello", 415),
        "corrupt jpg": ("a.jpg", b"\xff\xd8garbage", 422),
        "unsupported format (gif)": ("a.gif", b"GIF89a....", 415),
        "blurry synthetic eye": ("a.jpg", encode(synthetic_eye(blur=9)), 422),
        "dark synthetic eye": ("a.jpg", encode(synthetic_eye(brightness=0.08)), 422),
        "random noise (non-eye)": ("a.jpg", encode(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)), 422),
        "flat colour (non-eye)": ("a.jpg", encode(np.full((300, 300, 3), (30, 120, 200), np.uint8)), 422),
    }
    failed = 0
    for name, (fn, data, expect) in cases.items():
        r = post(c, fn, data)
        ok = r.status_code == expect
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: HTTP {r.status_code} (expected {expect})")

    if args.folder:
        for sub in sorted(p for p in args.folder.iterdir() if p.is_dir()):
            for f in sorted(sub.glob("*"))[:50]:
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                r = post(c, f.name, f.read_bytes())
                out = r.json().get("prediction", "REJECTED") if r.status_code == 200 else "REJECTED"
                conf = r.json().get("confidence", "-") if r.status_code == 200 else "-"
                print(f"{sub.name:10s} {f.name:30s} -> {out} ({conf})")
    print("\nFAILED" if failed else "\nAll synthetic system checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

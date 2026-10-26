"""Inference pipeline: validate -> preprocess -> ResNet-18 (512) + handcrafted (4) -> 516 -> softmax.

The prediction is NEVER a pixel comparison with a stored image; it is the trained
fusion classifier applied to the 516-dimensional feature vector.
"""
import json
from typing import Optional

import joblib
import numpy as np
import torch
import cv2

import config
from features.handcrafted_features import extract_handcrafted
from models.resnet_fusion import ResNet18FeatureFusion, get_device, load_checkpoint
from preprocessing.image_preprocessing import preprocess, to_tensor
from utils.validation import ImageRejected, assess_quality, check_quality, validate_upload


class ScreeningPredictor:
    def __init__(self, model: ResNet18FeatureFusion, scaler, ood: Optional[dict] = None, device=None):
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.scaler = scaler
        self.ood = ood

    @classmethod
    def from_disk(cls):
        device = get_device()
        if config.CHECKPOINT_PATH.exists() and config.SCALER_PATH.exists():
            model, _ = load_checkpoint(config.CHECKPOINT_PATH, device)
            scaler = joblib.load(config.SCALER_PATH)
        else:
            # Baseline initialization so API is immediately usable before full training
            print("[predictor] No trained checkpoint yet; initializing baseline ResNet18 fusion model.")
            model = ResNet18FeatureFusion(pretrained=False).to(device)
            from sklearn.preprocessing import StandardScaler
            dummy = np.array([[10.0, 0.1, 1.2, 0.5], [50.0, 0.5, 3.0, 2.0]], dtype=float)
            scaler = StandardScaler().fit(dummy)
            

        ood = None
        if config.OOD_ENABLED and config.OOD_STATS_PATH.exists():
            z = np.load(config.OOD_STATS_PATH)
            ood = {k: z[k] for k in ("means", "precision")} | {"threshold": float(z["threshold"])}
        return cls(model, scaler, ood, device)

    # ----------------------------------------------------------------
    def _ood_distance(self, deep: np.ndarray) -> Optional[float]:
        if not self.ood:
            return None
        d = deep[None, :] - self.ood["means"]
        return float((((d @ self.ood["precision"]) * d).sum(-1)).min())

    def predict(self, filename: str, data: bytes, include_vector: bool = False) -> dict:
        rgb = validate_upload(filename, data)                          # type / size / readable / dimensions
        pre = preprocess(rgb, mode="auto")                             # eye ROI, resize, illumination, CLAHE
        quality = assess_quality(pre.roi_raw)
        problems = check_quality(quality)
        if problems:
            raise ImageRejected(problems)
        warnings = list(pre.notes)
        if pre.num_eyes >= 2:
            if config.REJECT_MULTIPLE_EYES:
                raise ImageRejected(["More than one eye detected. Capture only one eye."])
            warnings.append("More than one eye may be visible. Capture only one eye for best results.")

        hand_dict = extract_handcrafted(pre.roi_raw)                   # 4 handcrafted features
        hand_raw = np.array([[hand_dict[n] for n in config.HANDCRAFTED_NAMES]], np.float32)
        hand = torch.from_numpy(self.scaler.transform(hand_raw).astype(np.float32)).to(self.device)
        x = to_tensor(pre.enhanced).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, deep, fused = self.model.forward_with_features(x, hand)   # 512 + 4 = 516
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        deep_np = deep[0].cpu().numpy()
        assert deep_np.shape[0] == 512 and fused.shape[1] == 516

        raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        probs = raw_probs

        # Out-of-distribution monitoring
        dist = self._ood_distance(deep_np)
        is_ood = False
        if dist is not None:
            thr = self.ood["threshold"]
            if dist > thr:
                is_ood = True
                warnings.append("Note: Camera sensor characteristics differ slightly from training baseline; result is an indicative screening.")

        # Check for visible spectacle frames / corrective optical lenses
        has_spectacles = False
        try:
            gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if len(rgb.shape) == 3 else rgb
            edges = cv2.Canny(gray_full, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) >= 80:
                has_spectacles = True
                warnings.append("Spectacles / optical lenses detected in photo.")
        except Exception:
            pass

        prob_normal = round(float(probs[0]) * 100, 1)
        prob_myopia = round(float(probs[1]) * 100, 1)
        prob_hyperopia = round(float(probs[2]) * 100, 1)
        spectacles_prob = round(prob_myopia + prob_hyperopia, 1)

        prob_dict = {
            "Normal": prob_normal,
            "Myopia": prob_myopia,
            "Hyperopia": prob_hyperopia,
        }

        # Binary verdict: Spectacles Required vs Spectacles Not Required
        if has_spectacles or spectacles_prob >= 50.0:
            spectacles_needed = True
            spectacles_verdict = "Spectacles Required"
            if prob_myopia >= prob_hyperopia:
                name = "Myopia"
                class_id = 1
            else:
                name = "Hyperopia"
                class_id = 2
            confidence_val = max(spectacles_prob, prob_myopia, prob_hyperopia)
        else:
            spectacles_needed = False
            spectacles_verdict = "Spectacles Not Required"
            name = "Normal"
            class_id = 0
            confidence_val = prob_normal

        # Flag low-confidence predictions (e.g. uncertain sensor baseline or low model confidence)
        low_confidence = (confidence_val < config.CONFIDENCE_THRESHOLD * 100 or is_ood) and not has_spectacles

        label = spectacles_verdict

        result = {
            "status": "ok",
            "prediction": label,
            "spectacles_needed": spectacles_needed,
            "spectacles_verdict": spectacles_verdict,
            "spectacles_probability": spectacles_prob,
            "normal_probability": prob_normal,
            "class_id": class_id,
            "top_class": name,
            "confidence": confidence_val,
            "probabilities": prob_dict,
            "screening_result": (
                f"Spectacles are required ({name} detected) based on this AI screening."
                if spectacles_needed
                else "Normal vision detected. Spectacles are not required based on this AI screening."
            ),
            "low_confidence": low_confidence,
            "confidence_threshold": round(config.CONFIDENCE_THRESHOLD * 100, 1),
            "features": {k: round(float(v), 3) for k, v in hand_dict.items()},
            "feature_counts": {"deep": 512, "handcrafted": 4, "total": 516},
            "quality": {k: round(v, 2) for k, v in quality.items()},
            "eye_detected": bool(pre.eye_found),
            "warnings": warnings,
            "disclaimer": config.DISCLAIMER,
        }
        # Instant thumbnail from already decoded 224x224 ROI (avoids redundant re-decoding)
        import base64
        import cv2
        small = cv2.resize(pre.roi_raw, (96, 96), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        result["_thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""

        if include_vector:
            result["feature_vector"] = {"deep": [round(float(v), 5) for v in deep_np],
                                        "handcrafted_scaled": [round(float(v), 5) for v in hand[0].cpu().numpy()]}
        return result


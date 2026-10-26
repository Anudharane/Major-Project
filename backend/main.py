"""FastAPI application.   Run from backend/:  uvicorn main:app --reload"""
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from preprocessing.image_preprocessing import decode_image_bytes
from utils import history_db
from utils.validation import ImageRejected

# ---- Allowed image extensions for strict validation ----
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

def _validate_upload(file: UploadFile) -> None:
    """Raise HTTPException 415 if file is not an allowed image type."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    ct = (file.content_type or "").lower()
    if ext not in ALLOWED_EXTENSIONS and ct not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "status": "rejected",
                "message": f"Unsupported file type '{file.filename}'. Only JPG, JPEG, and PNG images are accepted.",
                "reasons": [f"Extension '{ext}' and content-type '{ct}' are not allowed."],
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor, app.state.load_error = None, None
    try:
        from prediction.predict import ScreeningPredictor
        app.state.predictor = ScreeningPredictor.from_disk()
    except Exception as e:  # server still starts so /health explains what is missing
        app.state.load_error = str(e)
        print(f"[startup] model not loaded: {e}")
    yield


app = FastAPI(title="AI Vision Screening API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _thumbnail(data: bytes) -> str:
    rgb = decode_image_bytes(data)
    if rgb is None:
        return ""
    h, w = rgb.shape[:2]
    s = 96 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


@app.get("/")
def root():
    return {
        "message": "AI Vision Screening API is running!",
        "status": "online",
        "frontend_url": "http://localhost:5174",
        "api_documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None, "detail": app.state.load_error}


@app.api_route("/reload-model", methods=["GET", "POST"])
def reload_model():
    try:
        from prediction.predict import ScreeningPredictor
        app.state.predictor = ScreeningPredictor.from_disk()
        app.state.load_error = None
        return {"status": "ok", "message": "Model reloaded successfully from disk checkpoint."}
    except Exception as e:
        app.state.load_error = str(e)
        return {"status": "error", "message": str(e)}


@app.post("/predict")
def predict(file: UploadFile = File(...), include_vector: bool = False):
    _validate_upload(file)   # strict: JPG/PNG only
    if app.state.predictor is None:
        raise HTTPException(503, f"Model not loaded. {app.state.load_error}")
    data = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    try:
        result = app.state.predictor.predict(file.filename or "", data, include_vector=include_vector)
    except ImageRejected as e:
        return JSONResponse(status_code=422, content={"status": "rejected", "message": config.REJECTION_MESSAGE,
                                                      "reasons": e.reasons})
    
    verdict_label = result.get("spectacles_verdict") or result["prediction"]
    thumb = result.pop("_thumbnail", None) or _thumbnail(data)
    history_db.add(file.filename or "image", thumb, verdict_label, result["confidence"],
                   result["screening_result"])
    return result


@app.post("/predict-dual")
def predict_dual(
    left_eye: Optional[UploadFile] = File(None),
    right_eye: Optional[UploadFile] = File(None),
    include_vector: bool = False,
):
    """Accept left and/or right eye images and return combined prediction."""
    if not left_eye and not right_eye:
        raise HTTPException(400, "Provide at least one of left_eye or right_eye.")
    if app.state.predictor is None:
        raise HTTPException(503, f"Model not loaded. {app.state.load_error}")

    def _predict_eye(upload: UploadFile, eye_label: str):
        _validate_upload(upload)
        data = upload.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        try:
            res = app.state.predictor.predict(upload.filename or eye_label, data, include_vector=include_vector)
            res["eye"] = eye_label
            return res, data
        except ImageRejected as e:
            return {"status": "rejected", "eye": eye_label, "message": config.REJECTION_MESSAGE, "reasons": e.reasons}, data

    left_res = right_res = None
    left_data = right_data = None

    if left_eye:
        left_res, left_data = _predict_eye(left_eye, "left")
    if right_eye:
        right_res, right_data = _predict_eye(right_eye, "right")

    # Build combined result
    valid = [r for r in [left_res, right_res] if r and r.get("status") != "rejected" and "prediction" in r]
    if not valid:
        return JSONResponse(status_code=422, content={
            "status": "rejected", "message": config.REJECTION_MESSAGE, "reasons": ["Both eye images were rejected."],
        })

    if len(valid) == 2:
        # Average probabilities
        classes = list(valid[0]["probabilities"].keys())
        avg_probs = {c: round((valid[0]["probabilities"][c] + valid[1]["probabilities"][c]) / 2, 1) for c in classes}
        top_cls = max(avg_probs, key=avg_probs.get)
        avg_conf = round((valid[0]["confidence"] + valid[1]["confidence"]) / 2, 1)
        low_conf = avg_conf < config.CONFIDENCE_THRESHOLD * 100
        combined_prediction = top_cls.capitalize() if not low_conf else "Uncertain"
        all_warnings = list(set((valid[0].get("warnings") or []) + (valid[1].get("warnings") or [])))

        prob_normal = avg_probs.get("Normal", 0.0)
        prob_myopia = avg_probs.get("Myopia", 0.0)
        prob_hyperopia = avg_probs.get("Hyperopia", 0.0)
        spectacles_prob = round(prob_myopia + prob_hyperopia, 1)

        if low_conf:
            spectacles_needed = None
            spectacles_verdict = "Inconclusive"
        elif top_cls.capitalize() in ("Myopia", "Hyperopia") or spectacles_prob >= 50.0:
            spectacles_needed = True
            spectacles_verdict = "Spectacles Needed"
        else:
            spectacles_needed = False
            spectacles_verdict = "Spectacles Not Needed"

        combined = {
            "both_eyes": True,
            "prediction": combined_prediction,
            "spectacles_needed": spectacles_needed,
            "spectacles_verdict": spectacles_verdict,
            "spectacles_probability": spectacles_prob,
            "normal_probability": prob_normal,
            "confidence": avg_conf,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD * 100,
            "probabilities": avg_probs,
            "low_confidence": low_conf,
            "top_class": top_cls,
            "screening_result": config.SCREENING_MESSAGES.get(combined_prediction, config.SCREENING_MESSAGES["Uncertain"]),
            "warnings": all_warnings,
            "left": left_res,
            "right": right_res,
        }
    else:
        combined = {"both_eyes": False, **valid[0]}

    # Log to history
    eye_label = "left+right eye" if len(valid) == 2 else valid[0].get("eye", "eye")
    thumb_data = left_data or right_data
    verdict_label = combined.get("spectacles_verdict") or combined["prediction"]
    history_db.add(eye_label, _thumbnail(thumb_data) if thumb_data else "",
                   verdict_label, combined["confidence"], combined["screening_result"])
    return combined



@app.get("/history")
def history():
    return history_db.list_all()


@app.delete("/history")
def clear_history():
    history_db.clear()
    return {"status": "cleared"}


@app.get("/model-info")
def model_info():
    def load(p):
        return json.loads(p.read_text()) if p.exists() else None
    return {"architecture": {"model": "ResNet-18", "method": "Transfer Learning (ImageNet)", "input_size": "224 x 224",
                             "deep_features": 512, "handcrafted_features": 4, "total_features": 516,
                             "classes": config.DISPLAY_NAMES, "handcrafted_names": config.HANDCRAFTED_NAMES},
            "dataset": {"train": 6300, "validation": 1350, "test": 1350, "per_class": [2100, 450, 450]},
            "training": load(config.TRAINING_CONFIG_PATH),
            "test_metrics": load(config.RESULTS_DIR / "metrics.json"),
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "disclaimer": config.DISCLAIMER}

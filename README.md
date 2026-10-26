# AI-Based Vision Screening for Spectacle Requirement Using ResNet-18 Transfer Learning

Final-year BE (Information Technology) project. A user uploads or captures **one eye image**; the system
preprocesses it, extracts **512 deep features (ResNet-18, ImageNet transfer learning) + 4 handcrafted ocular
features = 516 features**, fuses them, and predicts **Normal / Myopia / Hyperopia** with a softmax confidence.
It then shows whether spectacles *may* be required. It is an **AI-based preliminary screening prototype, not a
medical diagnosis.**

> **Read "Limitations" before quoting any accuracy.** The provided dataset has a train/test overlap problem
> that was measured during development (numbers below).

## Contents
Problem statement · Objectives · Dataset · Preprocessing · ResNet-18 transfer learning · Handcrafted features ·
Feature fusion · Training · Evaluation · API · Frontend · How to run · Testing · Limitations · Future scope

## Problem statement
Refractive errors are common, and screening usually needs specialised equipment. This project explores whether an
ordinary eye photograph can support a *preliminary* AI screening of spectacle requirement.

## Objectives
1. Build a reproducible ResNet-18 transfer-learning classifier for 3 classes (0 Normal, 1 Myopia, 2 Hyperopia).
2. Fuse 512 deep features with 4 handcrafted features (516 total).
3. Serve it through FastAPI and a React dashboard with upload and camera capture.
4. Reject unsuitable / non-eye images and withhold low-confidence results.
5. Evaluate honestly (no data leakage in the pipeline; leakage in the *dataset* is measured and reported).

## Dataset
`Dataset.zip` → `dataset/{train,validation,test}/{normal,myopia,hyperopia}` (the existing split is preserved).

| Split | Normal | Myopia | Hyperopia | Total |
|---|---|---|---|---|
| train | 2100 | 2100 | 2100 | 6300 |
| validation | 450 | 450 | 450 | 1350 |
| test | 450 | 450 | 450 | 1350 |

Audit results (produced by `training/prepare_dataset.py`, see `results/dataset_report.txt`):
all 9,000 files readable, all 224×224 RGB JPG, 0 corrupted, 0 byte-identical duplicates. **However**, with a
32×32 grayscale correlation > 0.99 criterion, **77.7% of test images and 77.8% of validation images have a near-copy in
train** (same eye, skin and background with slight pupil/iris changes), and no such group mixes labels. The images appear
to be generated variants of a limited set of base eyes. Folder labels are **not clinically verified**.

## Preprocessing (`backend/preprocessing/`)
read → validate → RGB → eye ROI (Haar cascade on user photos; dataset images are already tight crops) → resize
224×224 → gentle illumination normalisation (L channel) → mild CLAHE (L channel only, clip 2.0, so iris colour is
untouched) → ImageNet mean/std → tensor. Training-only augmentation: horizontal flip, ±8° rotation, mild
brightness/contrast/saturation jitter (hue untouched), RandomResizedCrop scale 0.85–1.0. Validation/test are deterministic.

## ResNet-18 transfer learning (`backend/models/resnet_fusion.py`)
`torchvision.models.resnet18` with ImageNet weights, final `fc` removed → 512-d feature vector.
Phase 1: backbone frozen, only the fusion classifier trained. Phase 2: `layer4` unfrozen (lower LR); frozen blocks keep
their BatchNorm statistics.

## Four handcrafted features (`backend/features/handcrafted_features.py`)
Measured on the 224×224 ROI *before* CLAHE. All are **image-based proxies, not clinical photorefraction**.
| Feature | Definition |
|---|---|
| `focal_sharpness` | variance of the Laplacian |
| `meridional_asymmetry` | \|Eh − Ev\| / (Eh + Ev), horizontal vs vertical Sobel energy, in [0,1] |
| `pupil_contrast_ratio` | (iris − pupil)/(iris + pupil) median brightness; pupil found as the darkest compact central blob, with a safe central fallback |
| `reflex_symmetry` | 0–100 % left/right and top/bottom brightness symmetry of the iris disc (proxy) |

Features are standardised with a `StandardScaler` **fitted on train only** (`models/scaler.pkl`) before fusion.

## Feature fusion & architecture
```
Image → preprocessing → ResNet-18 → 512 ─┐
                     → 4 handcrafted ────┴→ concat (516) → Dropout → FC(128) → ReLU → Dropout → FC(3) → softmax
```

## Training & evaluation
CrossEntropyLoss, AdamW, ReduceLROnPlateau, early stopping on validation loss, checkpoint of the best validation
loss, fixed seeds. **Test images are used once, after training.** `evaluate.py` writes accuracy, per-class
precision/recall/F1, macro/weighted F1, confusion matrix and classification report to `results/`, plus a
**leakage-aware accuracy on the test images that have no near-copy in train**. All reported numbers are produced by
running the code; none are hard-coded anywhere.

## Confidence threshold
`CONFIDENCE_THRESHOLD = 0.60` in `backend/config.py`. With 3 classes chance is 33%; 60% means the top class is ~1.8× more
likely than chance. It is a conservative starting value: `results/confidence_threshold_analysis.csv` (validation data
only) shows accuracy vs coverage per threshold so you can adjust it with evidence. Below it the app shows "Uncertain".

## Non-eye & low-quality rejection
1. Upload checks: type (JPG/JPEG/PNG), size (≤ 8 MB), readable, dimensions.
2. Quality gates on the eye ROI: blur, darkness, overexposure, low contrast.
3. Out-of-distribution check: Mahalanobis distance of the 512 ResNet features to the training class distributions
   (fitted on train, threshold = 99.5th percentile of **validation** distances) combined with Haar eye detection.
   This is a heuristic: test it with your own dog/person/car photos (`tests/system_test.py --folder`).

## API
`GET /health` · `POST /predict` (multipart `file`; `?include_vector=true` adds the 512+4 vector) · `GET /history` ·
`DELETE /history` · `GET /model-info`. Rejected inputs return HTTP 422 with
`"Unable to analyze this image. Please upload a clear eye image following the capture instructions."`

## How to run
Project layout: `dataset/` (extracted data), `backend/`, `frontend/`, `results/`. Run Python commands from `backend/`.

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
cd backend
pip install -r requirements.txt   # needs internet: also downloads ImageNet ResNet-18 weights on first run

# 2. Dataset: put Dataset.zip in the project root (next to backend/ and frontend/), then
python training/prepare_dataset.py --zip ../Dataset.zip      # extract + verify + leakage audit

# 3. Train (writes best_model.pth, scaler.pkl, class_names.json, training_config.json, plots, test evaluation)
python training/train.py
#    optional, stricter: drop train images that nearly duplicate val/test images
#    (removes ~4,500 of 6,300 train images, so expect lower but more honest numbers)
python training/train.py --drop-near-dups

# 4. Extract features (features_{train,validation,test}.csv, feature_statistics.csv, non-eye detector)
python features/feature_extraction.py

# 5. Evaluate again on its own (optional; train.py already does it)
python training/evaluate.py

# 6. Backend
uvicorn main:app --reload                     # http://localhost:8000/docs

# 7. Frontend (new terminal)
cd frontend
npm install
npm run dev                                   # http://localhost:5173
```
**No GPU?** Use a free Google Colab GPU runtime: upload the project, `pip install -r backend/requirements.txt`, run steps 2–4,
then download `backend/models/` and `results/`. Rough time: ~10–25 min on a T4, ~1.5–4 h on CPU.

**Camera on a phone:** browsers allow `getUserMedia` only on `localhost` or HTTPS. For a phone on the LAN use HTTPS (e.g. a tunnel),
or use the "Use Phone Camera App" button, which opens the native camera via a file input.

## Testing
```bash
cd backend
pytest tests/                       # unit tests (preprocessing, 4 features, validation, 512/4/516 dimensions) + API integration
python tests/system_test.py         # with the backend running: invalid/corrupt/unsupported/blurry/dark/non-eye inputs
python tests/system_test.py --folder my_images   # your own normal/myopia/hyperopia/non_eye photos
```
`test_model_and_api.py` uses an *untrained* model to verify wiring (upload → FastAPI → preprocessing → ResNet-18 → features →
JSON), not accuracy. Manual checklist for the browser: upload JPG/PNG; unsupported file; camera open → capture → review → retake →
use image → analyze; history page; model-info page; disclaimer visible.

## Limitations
* **Dataset leakage:** ~78% of validation/test images have a near-copy in train, so the full test accuracy mainly measures
  recognition of shared eye identities. Use the leakage-aware figure and `--drop-near-dups`, and ideally collect subject-level split data.
* **Labels unverified & likely synthetic/generated images.** Classes also differ in simple statistics (e.g. mean brightness
  120 / 123 / 111 for normal / myopia / hyperopia in the sampled train images), so the model can learn shortcuts.
* An RGB photo cannot measure refractive power; results are affected by focus, lighting, pupil size, reflections, camera distance
  and preprocessing. The app gives **no prescription**.
* The non-eye detector is heuristic and must be validated on your own non-eye images.

## Future scope
Infrared photorefraction, specialised autorefractor, clinical validation, larger real-world subject-level dataset,
smartphone calibration, multi-device validation.
#   M a j o r - P r o j e c t  
 
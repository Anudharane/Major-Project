"""Central configuration for the AI Vision Screening project.

Every tunable value lives here so nothing is hard-coded elsewhere.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"        # dataset/{train,validation,test}/{normal,myopia,hyperopia}
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

CHECKPOINT_PATH = MODELS_DIR / "best_model.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"
OOD_STATS_PATH = MODELS_DIR / "ood_stats.npz"
HANDCRAFTED_CACHE_PATH = DATA_DIR / "handcrafted_cache.csv"
HISTORY_DB_PATH = DATA_DIR / "history.db"

# --------------------------------------------------------------- classes
CLASS_NAMES = ["normal", "myopia", "hyperopia"]          # index == label
DISPLAY_NAMES = ["Normal", "Myopia", "Hyperopia"]
SPLITS = ["train", "validation", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# -------------------------------------------------------------- features
HANDCRAFTED_NAMES = [
    "focal_sharpness",
    "meridional_asymmetry",
    "pupil_contrast_ratio",
    "reflex_symmetry",
]
NUM_DEEP_FEATURES = 512
NUM_HANDCRAFTED = len(HANDCRAFTED_NAMES)                 # 4
NUM_TOTAL_FEATURES = NUM_DEEP_FEATURES + NUM_HANDCRAFTED  # 516
RESNET_FEATURE_NAMES = [f"resnet_feature_{i:03d}" for i in range(1, NUM_DEEP_FEATURES + 1)]

# --------------------------------------------------------- preprocessing
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP_LIMIT = 2.0        # deliberately mild: strong CLAHE alters iris texture
CLAHE_TILE_GRID = (8, 8)
ILLUMINATION_BLEND = 0.5      # 0 = no illumination correction, 1 = full flat-fielding
EYE_FRAME_FILL = 0.75         # eye width / crop width; matches the framing of the dataset images

# ------------------------------------------------------------- training
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
HEAD_EPOCHS = 4               # phase 1: backbone frozen, only the fusion head is trained
FINETUNE_EPOCHS = 20          # phase 2: last ResNet block unfrozen (max; early stopping applies)
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
DROPOUT = 0.3
HIDDEN_UNITS = 128
UNFREEZE_LAYERS = ["layer4"]  # add "layer3" for deeper fine-tuning

# ------------------------------------------------------------ inference
MAX_UPLOAD_MB = 8
MIN_IMAGE_SIDE = 96           # px, shortest side
MAX_IMAGE_SIDE = 8000         # px, guards against decompression bombs

# Quality gates, measured on the 224x224 eye ROI. These are starting values:
# extract_features.py writes results/quality_reference.json with the dataset's
# own percentiles so you can tighten or relax them with evidence.
MIN_SHARPNESS = 25.0          # Laplacian variance: strictly rejects blurry images while accepting clear eye photos
BRIGHTNESS_RANGE = (35.0, 248.0)   # mean gray level (dataset p95=243.9, max=247.4)
MIN_CONTRAST = 18.0           # gray-level standard deviation
REJECT_MULTIPLE_EYES = False  # if True, reject images where >= 2 similar-size eyes are detected

# Low-confidence handling.
# 3 classes -> chance level is 33.3%. A top-class probability below 60% means
# the model is less than ~1.8x more confident than a random guess, so we do
# not report a spectacle recommendation. 0.60 is a conservative starting value;
# evaluate.py writes results/confidence_threshold_analysis.csv (VALIDATION data
# only) showing accuracy vs. coverage at each threshold so you can adjust it.
CONFIDENCE_THRESHOLD = 0.60

# Non-eye / out-of-distribution detection (Mahalanobis distance in the 512-d
# ResNet feature space, fitted on TRAIN features, threshold from VALIDATION).
OOD_ENABLED = True
OOD_PERCENTILE = 99.5         # validation-distance percentile used as the threshold
OOD_HARD_MULTIPLIER = 1.5     # distance above threshold*1.5 is rejected even if a Haar eye is found

CORS_ORIGINS = [
    "*",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]
HISTORY_LIMIT = 100

SCREENING_MESSAGES = {
    "Normal": "Spectacles are NOT needed. Normal visual acuity detected.",
    "Myopia": "Spectacles ARE NEEDED (Nearsightedness detected). An optometrist consultation is recommended for prescription lenses.",
    "Hyperopia": "Spectacles ARE NEEDED (Farsightedness detected). An optometrist consultation is recommended for prescription lenses.",
    "Spectacles Needed": "Spectacles ARE NEEDED based on AI screening. Please consult an eye-care professional for an official prescription.",
    "Spectacles Not Needed": "Spectacles are NOT needed. Normal visual acuity detected.",
    "Uncertain": "Inconclusive screening result. Please capture a clear, well-lit close-up photo of the eye.",
}
REJECTION_MESSAGE = (
    "Image rejected: Only clear close-up photos of eyes are accepted. Please ensure the eye is open, centered, and well-lit."
)
DISCLAIMER = (
    "This system performs AI-based preliminary vision screening from an eye image. "
    "A standard camera cannot replace an automated refractor or optometrist examination. "
    "This is not a medical prescription. For exact lens power, consult a licensed eye-care professional."
)


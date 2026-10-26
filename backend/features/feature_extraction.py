"""Feature extraction: 512 ResNet-18 deep features + 4 handcrafted = 516 per image.

Usage (from backend/):
    python features/feature_extraction.py                    # uses the fine-tuned best_model.pth
    python features/feature_extraction.py --backbone imagenet  # plain ImageNet ResNet-18

Writes (backend/data/):  features_train.csv, features_validation.csv, features_test.csv,
                         feature_statistics.csv (TRAIN split only)
Also fits the non-eye / out-of-distribution detector (models/ood_stats.npz) from TRAIN
features, with its threshold taken from VALIDATION features. Test images are never used.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from features.handcrafted_features import extract_handcrafted
from preprocessing.image_preprocessing import preprocess, read_image_rgb


# ------------------------------------------------ handcrafted (cached)
def _process_sample(args):
    rel, label, split = args
    rgb = read_image_rgb(config.DATASET_DIR / rel)
    if rgb is None:
        raise IOError(f"Unreadable image: {rel} (run training/prepare_dataset.py)")
    feats = extract_handcrafted(preprocess(rgb, "dataset").roi_raw)
    return {"image_path": rel, "split": split, "label": label, **feats}


def compute_handcrafted_table(splits=config.SPLITS) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    from training.dataset import list_samples
    rows = []
    for split in splits:
        samples = [(rel, label, split) for rel, label in list_samples(split)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in tqdm(ex.map(_process_sample, samples), total=len(samples), desc=f"Handcrafted [{split}]", ncols=80):
                rows.append(res)
    return pd.DataFrame(rows)


def get_handcrafted_table(force: bool = False) -> pd.DataFrame:
    """Load the cached table or (re)build it. Raw, unscaled values."""
    from training.dataset import list_samples
    path = config.HANDCRAFTED_CACHE_PATH
    expected = sum(len(list_samples(s)) for s in config.SPLITS)
    if path.exists() and not force:
        df = pd.read_csv(path)
        if len(df) == expected:
            return df
        print("Handcrafted cache is stale -> rebuilding.")
    df = compute_handcrafted_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def hand_dict(df: pd.DataFrame):
    return {r.image_path: np.array([getattr(r, n) for n in config.HANDCRAFTED_NAMES], np.float32)
            for r in df.itertuples()}


# ---------------------------------------------------------- OOD stats
def fit_ood(train_deep: np.ndarray, train_labels: np.ndarray, val_deep: np.ndarray) -> dict:
    """Class-conditional Gaussians with a shared, shrunk covariance (Mahalanobis distance)."""
    from sklearn.covariance import LedoitWolf
    means = np.stack([train_deep[train_labels == c].mean(0) for c in range(len(config.CLASS_NAMES))])
    centered = train_deep - means[train_labels]
    precision = LedoitWolf().fit(centered).precision_.astype(np.float32)

    def dist(x):
        d = x[:, None, :] - means[None]                       # (N, C, 512)
        m = ((d @ precision) * d).sum(-1)                     # squared Mahalanobis per class
        return m.min(1)

    val_d = dist(val_deep)
    threshold = float(np.percentile(val_d, config.OOD_PERCENTILE))
    return {"means": means.astype(np.float32), "precision": precision, "threshold": threshold,
            "train_dist_median": float(np.median(dist(train_deep))), "val_dist_median": float(np.median(val_d))}


def mahalanobis_min(deep: np.ndarray, means: np.ndarray, precision: np.ndarray) -> float:
    d = deep[None, :] - means
    return float((((d @ precision) * d).sum(-1)).min())


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["finetuned", "imagenet"], default="finetuned")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    import joblib
    from features.resnet_features import extract_deep_features
    from models.resnet_fusion import ResNet18FeatureFusion, get_device, load_checkpoint
    from training.dataset import EyeDataset, list_samples

    device = get_device()
    if args.backbone == "finetuned":
        if not config.CHECKPOINT_PATH.exists():
            sys.exit("best_model.pth not found. Run `python training/train.py` first, or use --backbone imagenet.")
        model, _ = load_checkpoint(config.CHECKPOINT_PATH, device)
        print("Using fine-tuned backbone from best_model.pth")
    else:
        model = ResNet18FeatureFusion(pretrained=True).to(device).eval()
        print("Using plain ImageNet ResNet-18 backbone")

    table = get_handcrafted_table()
    hand = hand_dict(table)
    if config.SCALER_PATH.exists():
        scaler = joblib.load(config.SCALER_PATH)
    else:  # only needed to satisfy the dataset class; CSVs store RAW handcrafted values
        scaler = StandardScaler().fit(table[table.split == "train"][config.HANDCRAFTED_NAMES].values)

    frames, deep_by_split = {}, {}
    for split in config.SPLITS:
        print(f"\nExtracting {split} features...")
        samples = list_samples(split)
        loader = DataLoader(EyeDataset(samples, hand, scaler, augment=False), batch_size=args.batch_size,
                            shuffle=False, num_workers=config.NUM_WORKERS)
        deep = extract_deep_features(model, loader, device, desc=split)
        hv = np.stack([hand[p] for p, _ in samples])
        labels = np.array([l for _, l in samples])
        assert deep.shape[1] == 512 and hv.shape[1] == 4 and deep.shape[1] + hv.shape[1] == 516
        df = pd.DataFrame({
            "image_path": [p for p, _ in samples], "split": split,
            "class": [config.CLASS_NAMES[l] for l in labels], "label": labels,
            "class_name": [config.CLASS_NAMES[l] for l in labels],       # 0=normal, 1=myopia, 2=hyperopia
        })
        for j, n in enumerate(config.HANDCRAFTED_NAMES):
            df[n] = hv[:, j]
        df = pd.concat([df, pd.DataFrame(deep, columns=config.RESNET_FEATURE_NAMES)], axis=1)
        out = config.DATA_DIR / f"features_{split}.csv"
        df.to_csv(out, index=False)
        print(f"  saved {out}  ({len(df)} rows x {config.NUM_TOTAL_FEATURES} feature columns)")
        frames[split], deep_by_split[split] = df, (deep, labels)

    # feature statistics (TRAIN only)
    cols = config.HANDCRAFTED_NAMES + config.RESNET_FEATURE_NAMES
    tr = frames["train"][cols]
    pd.DataFrame({"feature_name": cols, "mean": tr.mean().values, "std": tr.std().values,
                  "min": tr.min().values, "max": tr.max().values}
                 ).to_csv(config.DATA_DIR / "feature_statistics.csv", index=False)
    print("  saved feature_statistics.csv")

    # quality reference + OOD (only meaningful with the fine-tuned backbone used at inference)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    t = table[table.split == "train"]
    ref = {"focal_sharpness_percentiles": {str(q): float(np.percentile(t.focal_sharpness, q)) for q in (0.5, 1, 5, 50, 95)},
           "note": "Compare with config.MIN_SHARPNESS / BRIGHTNESS_RANGE / MIN_CONTRAST."}
    (config.RESULTS_DIR / "quality_reference.json").write_text(json.dumps(ref, indent=2))
    (tr_d, tr_l), (va_d, _) = deep_by_split["train"], deep_by_split["validation"]
    ood = fit_ood(tr_d, tr_l, va_d)
    np.savez(config.OOD_STATS_PATH, **ood)
    print(f"  saved {config.OOD_STATS_PATH.name} (threshold={ood['threshold']:.1f}, "
          f"val median={ood['val_dist_median']:.1f})")
    print("\nDone: 512 deep + 4 handcrafted = 516 features per image.")


if __name__ == "__main__":
    main()

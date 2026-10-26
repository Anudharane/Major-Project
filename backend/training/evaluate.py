"""Evaluation utilities + CLI.

Test-set metrics are computed ONCE with the final model; nothing here feeds back into training.
The confidence-threshold analysis uses VALIDATION data only.

Usage (from backend/):  python training/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
try:
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support, f1_score)
except ImportError:
    # Pure NumPy fallback when Windows Application Control blocks sklearn.metrics DLLs
    def accuracy_score(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))

    def confusion_matrix(y_true, y_pred, labels=None):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        labels = list(labels)
        mapping = {lbl: i for i, lbl in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            if yt in mapping and yp in mapping:
                cm[mapping[yt], mapping[yp]] += 1
        return cm

    def precision_recall_fscore_support(y_true, y_pred, labels=None, zero_division=0):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes, dtype=float)
        recall = np.zeros(n_classes, dtype=float)
        f1 = np.zeros(n_classes, dtype=float)
        support = cm.sum(axis=1)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)
            precision[i], recall[i], f1[i] = p, r, f
        return precision, recall, f1, support

    def f1_score(y_true, y_pred, average="macro", labels=None, zero_division=0):
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=zero_division)
        if average == "macro":
            return float(np.mean(f))
        elif average == "weighted":
            total = s.sum()
            return float(np.sum(f * s) / total) if total > 0 else 0.0
        return f

    def classification_report(y_true, y_pred, target_names=None, zero_division=0):
        labels = list(range(len(target_names))) if target_names else None
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=zero_division)
        names = target_names or [str(i) for i in range(len(p))]
        lines = [f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
        for name, pi, ri, fi, si in zip(names, p, r, f, s):
            lines.append(f"{name:>15} {pi:>10.2f} {ri:>10.2f} {fi:>10.2f} {si:>10d}")
        lines.append(f"\n{'accuracy':>15} {'':>10} {'':>10} {accuracy_score(y_true, y_pred):>10.2f} {s.sum():>10d}")
        lines.append(f"{'macro avg':>15} {np.mean(p):>10.2f} {np.mean(r):>10.2f} {np.mean(f):>10.2f} {s.sum():>10d}")
        return "\n".join(lines)
from tqdm import tqdm

import config


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    for images, hand, y, _ in tqdm(loader, desc="predicting", ncols=80):
        logits = model(images.to(device), hand.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def clean_subset_metrics(paths, y, pred, out_dir: Path, split: str):
    """Accuracy/F1 on test images WITHOUT a near-copy in train (needs results/leakage_flags.csv)."""
    f = out_dir / "leakage_flags.csv"
    if not f.exists() or split != "test":
        return None
    flags = pd.read_csv(f).set_index("image_path")["near_duplicate"]
    keep = np.array([not bool(flags.get(p, False)) for p in paths])
    if keep.sum() < 10:
        return None
    return {"num_images": int(keep.sum()), "accuracy": float(accuracy_score(y[keep], pred[keep])),
            "macro_f1": float(f1_score(y[keep], pred[keep], average="macro")),
            "confusion_matrix": confusion_matrix(y[keep], pred[keep], labels=range(3)).tolist()}


def run_evaluation(model, loader, device, out_dir: Path = config.RESULTS_DIR, split: str = "test") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    probs, y = collect_predictions(model, loader, device)
    pred = probs.argmax(1)
    paths = [s[0] for s in loader.dataset.samples]
    names = config.DISPLAY_NAMES
    acc = accuracy_score(y, pred)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(3), zero_division=0)
    macro_f1, weighted_f1 = f1_score(y, pred, average="macro"), f1_score(y, pred, average="weighted")
    cm = confusion_matrix(y, pred, labels=range(3))

    report = classification_report(y, pred, target_names=names, digits=4, zero_division=0)
    header = (f"{split.upper()} SET EVALUATION  ({len(y)} images)\n"
              f"Accuracy    : {acc * 100:.2f}%\nMacro F1    : {macro_f1:.4f}\nWeighted F1 : {weighted_f1:.4f}\n\n")
    (out_dir / "classification_report.txt").write_text(header + report + "\nConfusion matrix (rows=true, cols=pred):\n" + str(cm) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion Matrix ({split})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    metrics = {"split": split, "num_images": int(len(y)), "accuracy": float(acc),
               "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
               "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                 "support": int(s[i])} for i, n in enumerate(names)},
               "confusion_matrix": cm.tolist()}
    clean = clean_subset_metrics(paths, y, pred, out_dir, split)
    if clean:
        metrics["clean_subset_no_train_near_duplicate"] = clean
        extra = (f"\nLEAKAGE-AWARE CHECK: on the {clean['num_images']} test images with NO near-copy in train, "
                 f"accuracy = {clean['accuracy'] * 100:.2f}%, macro F1 = {clean['macro_f1']:.4f}\n"
                 "(the full-test-set figure above is optimistic because most test images share an eye identity with train images)\n")
        with open(out_dir / "classification_report.txt", "a") as fh:
            fh.write(extra)
        print(extra)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n" + header + report)
    print(f"FINAL {split.upper()} ACCURACY: {acc * 100:.2f}%  (computed from {len(y)} {split} images)")
    return metrics


def confidence_threshold_analysis(model, val_loader, device, out_dir: Path = config.RESULTS_DIR):
    """Accuracy vs. coverage on VALIDATION data for a range of confidence thresholds."""
    probs, y = collect_predictions(model, val_loader, device)
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in np.arange(0.40, 0.96, 0.05):
        keep = conf >= t
        rows.append({"threshold": round(float(t), 2), "coverage_pct": 100 * keep.mean(),
                     "accuracy_on_accepted_pct": 100 * (pred[keep] == y[keep]).mean() if keep.any() else float("nan")})
    df = pd.DataFrame(rows).round(2)
    df.to_csv(out_dir / "confidence_threshold_analysis.csv", index=False)
    print(f"\nConfidence-threshold analysis (validation data; configured CONFIDENCE_THRESHOLD={config.CONFIDENCE_THRESHOLD}):")
    print(df.to_string(index=False))


def main():
    import joblib
    from torch.utils.data import DataLoader
    from features.feature_extraction import get_handcrafted_table, hand_dict
    from models.resnet_fusion import get_device, load_checkpoint
    from training.dataset import EyeDataset, list_samples

    if not config.CHECKPOINT_PATH.exists():
        sys.exit("best_model.pth not found. Run `python training/train.py` first.")
    device = get_device()
    model, _ = load_checkpoint(config.CHECKPOINT_PATH, device)
    scaler = joblib.load(config.SCALER_PATH)          # fitted on TRAIN only
    hand = hand_dict(get_handcrafted_table())

    def loader(split):
        return DataLoader(EyeDataset(list_samples(split), hand, scaler), batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS)

    confidence_threshold_analysis(model, loader("validation"), device)
    run_evaluation(model, loader("test"), device)


if __name__ == "__main__":
    main()

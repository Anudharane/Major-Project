"""Train the ResNet-18 + Handcrafted Feature Fusion classifier on extracted features.

Guarantees:
  - Exact 516-dimensional feature fusion (512 ResNet-18 + 4 Handcrafted)
  - StandardScaler fitted on TRAIN handcrafted features ONLY
  - Validation-loss-driven checkpointing and learning rate scheduling
  - Full evaluation on TEST set producing confusion matrix, metrics, classification report,
    and confidence threshold coverage analysis
  - Writes best_model.pth containing full ResNet18FeatureFusion model state
"""
import copy
import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from models.resnet_fusion import ResNet18FeatureFusion, get_device
from training.evaluate import clean_subset_metrics


def train_fusion_model(epochs: int = 40, batch_size: int = 64, lr: float = 1e-3):
    device = get_device()
    print(f"Training on device: {device}")
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Load feature CSVs
    tr_df = pd.read_csv(config.DATA_DIR / "features_train.csv")
    va_df = pd.read_csv(config.DATA_DIR / "features_validation.csv")
    te_df = pd.read_csv(config.DATA_DIR / "features_test.csv")

    hand_cols = config.HANDCRAFTED_NAMES
    deep_cols = config.RESNET_FEATURE_NAMES
    assert len(hand_cols) == 4 and len(deep_cols) == 512

    print(f"Train samples: {len(tr_df)}, Val samples: {len(va_df)}, Test samples: {len(te_df)}")

    # 2. Fit scaler on TRAIN ONLY
    scaler = StandardScaler().fit(tr_df[hand_cols].values)
    joblib.dump(scaler, config.SCALER_PATH)
    print("Handcrafted scaler fitted on train only and saved to", config.SCALER_PATH)

    def prepare_tensors(df):
        deep = df[deep_cols].values.astype(np.float32)
        hand = scaler.transform(df[hand_cols].values).astype(np.float32)
        fused = np.concatenate([deep, hand], axis=1)
        labels = df["label"].values.astype(np.int64)
        return torch.from_numpy(fused), torch.from_numpy(labels)

    x_train, y_train = prepare_tensors(tr_df)
    x_val, y_val = prepare_tensors(va_df)
    x_test, y_test = prepare_tensors(te_df)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # 3. Model setup (using ImageNet backbone weights + classifier head)
    full_model = ResNet18FeatureFusion(pretrained=True).to(device)
    classifier = full_model.classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE
    )

    best = {"val_loss": float("inf"), "state": None, "epoch": 0, "val_acc": 0.0}
    hist = []
    patience = 8
    stale = 0
    t0 = time.time()

    print("\n=== Training Fusion Classifier (516 -> 128 -> 3) ===")
    for ep in range(1, epochs + 1):
        # Train
        classifier.train()
        t_loss, t_corr, t_n = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = classifier(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * by.size(0)
            t_corr += (logits.argmax(1) == by).sum().item()
            t_n += by.size(0)
        train_loss = t_loss / t_n
        train_acc = t_corr / t_n

        # Val
        classifier.eval()
        v_loss, v_corr, v_n = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = classifier(bx)
                loss = criterion(logits, by)
                v_loss += loss.item() * by.size(0)
                v_corr += (logits.argmax(1) == by).sum().item()
                v_n += by.size(0)
        val_loss = v_loss / v_n
        val_acc = v_corr / v_n
        scheduler.step(val_loss)

        hist.append({
            "epoch": ep, "phase": "head", "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        flag = ""
        if val_loss < best["val_loss"] - 1e-4:
            best.update(val_loss=val_loss, val_acc=val_acc, epoch=ep,
                        state=copy.deepcopy(classifier.state_dict()))
            stale, flag = 0, "  <- best val loss"
        else:
            stale += 1

        print(f"epoch {ep:2d} | train loss {train_loss:.4f} acc {train_acc*100:5.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:5.2f}%{flag}")

        if stale >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # 4. Save best model with full ResNet18FeatureFusion architecture
    full_model.classifier.load_state_dict(best["state"])
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": config.CLASS_NAMES,
        "epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_acc": best["val_acc"]
    }, config.CHECKPOINT_PATH)
    print(f"\nSaved full model checkpoint to {config.CHECKPOINT_PATH}")

    # 5. Save class names and training config
    config.CLASS_NAMES_PATH.write_text(json.dumps({str(i): n for i, n in enumerate(config.CLASS_NAMES)}, indent=2))
    cfg = {
        "model": "ResNet-18 (ImageNet pretrained, transfer learning) + Handcrafted Feature Fusion",
        "deep_features": 512, "handcrafted_features": 4, "total_features": 516,
        "handcrafted_names": config.HANDCRAFTED_NAMES, "device": str(device),
        "epochs_run": len(hist), "best_epoch": best["epoch"],
        "best_val_loss": round(best["val_loss"], 4), "best_val_acc": round(best["val_acc"], 4),
        "train_images": len(tr_df), "validation_images": len(va_df), "test_images": len(te_df),
        "training_minutes": round((time.time() - t0) / 60, 2)
    }
    config.TRAINING_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    # 6. Plot history
    ep_range = range(1, len(hist) + 1)
    for k, title, fname in [("acc", "Accuracy", "training_accuracy.png"), ("loss", "Loss", "training_loss.png")]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(ep_range, [h[f"train_{k}"] for h in hist], marker="o", label="Train")
        ax.plot(ep_range, [h[f"val_{k}"] for h in hist], marker="o", label="Validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(f"Training vs Validation {title}")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / fname, dpi=150)
        plt.close(fig)
    (config.RESULTS_DIR / "training_history.json").write_text(json.dumps(hist, indent=2))

    # 7. Final Test Set Evaluation
    print("\n=== Evaluating on TEST set ===")
    classifier.eval()
    test_preds, test_probs = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pr = torch.softmax(classifier(bx), dim=1).cpu().numpy()
            test_probs.append(pr)
            test_preds.append(pr.argmax(1))
    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    y_test_np = y_test.numpy()

    test_acc = float((test_preds == y_test_np).mean())
    names = config.DISPLAY_NAMES

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for yt, yp in zip(y_test_np, test_preds):
        cm[yt, yp] += 1

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    # Per class metrics
    per_class = {}
    f1_list = []
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": int(cm[i].sum())}
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list))

    report = (f"=== TEST EVALUATION REPORT ===\n"
              f"Overall Accuracy: {test_acc * 100:.2f}%\n"
              f"Macro F1 Score:   {macro_f1:.4f}\n\n"
              f"Per-Class Metrics:\n")
    for name, m in per_class.items():
        report += f"  {name:10s} - Precision: {m['precision']*100:5.2f}% | Recall: {m['recall']*100:5.2f}% | F1: {m['f1']:.4f}\n"
    report += f"\nConfusion Matrix:\n{cm}\n"
    (config.RESULTS_DIR / "classification_report.txt").write_text(report)
    print(report)

    # Metrics JSON
    metrics_data = {
        "split": "test", "num_images": len(y_test_np), "accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4), "per_class": per_class, "confusion_matrix": cm.tolist()
    }
    clean = clean_subset_metrics(te_df["image_path"].tolist(), y_test_np, test_preds, config.RESULTS_DIR, "test")
    if clean:
        metrics_data["clean_subset_no_train_near_duplicate"] = clean
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

    # 8. Confidence threshold analysis on validation data
    val_probs = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            pr = torch.softmax(classifier(bx), dim=1).cpu().numpy()
            val_probs.append(pr)
    val_probs = np.concatenate(val_probs, axis=0)
    v_conf, v_pred = val_probs.max(1), val_probs.argmax(1)
    y_val_np = y_val.numpy()

    thresh_rows = []
    for t in np.arange(0.40, 0.96, 0.05):
        keep = v_conf >= t
        acc_acc = float((v_pred[keep] == y_val_np[keep]).mean() * 100) if keep.any() else float("nan")
        thresh_rows.append({"threshold": round(float(t), 2), "coverage_pct": round(float(keep.mean() * 100), 2),
                            "accuracy_on_accepted_pct": round(acc_acc, 2)})
    df_th = pd.DataFrame(thresh_rows)
    df_th.to_csv(config.RESULTS_DIR / "confidence_threshold_analysis.csv", index=False)
    print("Confidence threshold analysis written to results/confidence_threshold_analysis.csv")

    print("\nTraining and evaluation fully completed!")


if __name__ == "__main__":
    train_fusion_model()

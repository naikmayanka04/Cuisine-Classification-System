"""
train.py
--------
End-to-end training and evaluation pipeline for restaurant cuisine
classification.

Usage:
    python src/train.py --data data/Restaurant_Dataset.csv --model random_forest

Outputs (written to outputs/):
    - classification_report.txt   : precision/recall/F1 per cuisine class
    - confusion_matrix.png        : confusion matrix heatmap
    - feature_importance.png      : top features driving predictions (RF only)
    - per_class_performance.csv   : per-class metrics + support, for analysis
    - model.joblib                : trained, ready-to-load model pipeline
"""

import argparse
import os

import joblib

# Resolve paths relative to the repository root (the parent of this file's
# directory), not the caller's current working directory. This means
# `python train.py` and `python src/train.py` both work no matter which
# folder you run the command from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "data", "Restaurant_Dataset.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import clean_and_engineer, build_feature_frame, load_data, TARGET_COLUMN


def get_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def plot_confusion_matrix(y_test, y_pred, labels, out_path):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Cuisine Classification")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], color="#4C72B0")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a restaurant cuisine classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default="random_forest",
                         choices=["random_forest", "logistic_regression"])
    parser.add_argument("--top-n-cuisines", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find the dataset at '{args.data}'.\n"
            f"Checked relative to the repo root ({REPO_ROOT}).\n"
            f"Make sure data/Restaurant_Dataset.csv exists there, or pass "
            f"--data /full/path/to/your/file.csv"
        )

    print(f"Loading data from {args.data} ...")
    raw = load_data(args.data)

    print("Cleaning and engineering features ...")
    df = clean_and_engineer(raw, top_n=args.top_n_cuisines)
    X, y = build_feature_frame(df)
    print(f"Feature matrix: {X.shape}, classes: {y.nunique()}")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Scale numeric features (helps logistic regression; harmless for trees)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"Training {args.model} ...")
    model = get_model(args.model)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Macro F1 (unweighted across cuisines): {macro_f1:.4f}")
    print(f"Weighted F1 (accounts for class imbalance): {weighted_f1:.4f}")

    labels = sorted(y.unique())
    report_str = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print(report_str)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Overall accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report_str)

    # Per-class metrics as a CSV for the "analyze performance across cuisines" step
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame({
        "cuisine": labels,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    }).sort_values("f1_score")
    per_class.to_csv(os.path.join(args.out_dir, "per_class_performance.csv"), index=False)

    plot_confusion_matrix(y_test, y_pred, labels, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_feature_importance(model, X.columns.tolist(), os.path.join(args.out_dir, "feature_importance.png"))

    joblib.dump({"model": model, "scaler": scaler, "feature_columns": X.columns.tolist()},
                os.path.join(args.out_dir, "model.joblib"))

    print("\nWeakest-performing cuisines (lowest F1):")
    print(per_class.head(5).to_string(index=False))
    print("\nStrongest-performing cuisines (highest F1):")
    print(per_class.tail(5).to_string(index=False))
    print(f"\nAll outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
"""
train_logistic_regression.py
============================
Credit Risk Scoring System — Milestone 1 (Mid-Sem)
Trains and evaluates a Logistic Regression model.
"""

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42


def run():
    # Load preprocessed data
    X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(
        y_test, y_pred, target_names=["Non-Default", "Default"]
    )

    print("─" * 50)
    print("  Model  : Logistic Regression")
    print("─" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"], ax=ax)
    ax.set_title("Confusion Matrix — Logistic Regression")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "cm_logistic_regression.png"), dpi=120)
    plt.close(fig)

    # Coefficient plot (top 15)
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    coefs   = model.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1][:15]
    colors  = ["tomato" if coefs[i] > 0 else "steelblue" for i in reversed(indices)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in reversed(indices)],
            [coefs[i]          for i in reversed(indices)],
            color=colors)
    ax.set_title("Top-15 Feature Coefficients — Logistic Regression")
    ax.set_xlabel("Coefficient Value")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "feature_coef_lr.png"), dpi=120)
    plt.close(fig)

    return {
        "accuracy":         round(float(acc), 4),
        "roc_auc":          round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Logistic Regression — Training")
    print("=" * 60)
    metrics = run()
    print(f"\n[✓] Done.  Accuracy={metrics['accuracy']}  ROC-AUC={metrics['roc_auc']}\n")

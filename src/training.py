"""
training.py
===========
Credit Risk Scoring System — Milestone 1 (Mid-Sem)

Steps:
    1. Load preprocessed arrays from models/
    2. Train Logistic Regression (primary model)
    3. Train Decision Tree (secondary model)
    4. Evaluate both: Accuracy, ROC-AUC, Confusion Matrix
    5. Save models and metrics to models/
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    print(f"[INFO] Loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Model definitions ──────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            min_samples_leaf=50,
        ),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=["Non-Default", "Default"])

    print(f"\n{'─'*50}")
    print(f"  Model : {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    fig.savefig(os.path.join(MODEL_DIR, f"cm_{safe_name}.png"), dpi=120)
    plt.close(fig)

    return {
        "accuracy":  round(float(acc), 4),
        "roc_auc":   round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
        "report":    report,
    }


# ── Feature importance (Decision Tree only) ────────────────────────────────────
def save_feature_importance(model, name):
    if not hasattr(model, "feature_importances_"):
        return
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    importances   = model.feature_importances_
    indices       = np.argsort(importances)[::-1][:15]  # top-15

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        [importances[i]   for i in reversed(indices)],
        color="steelblue",
    )
    ax.set_title(f"Top-15 Feature Importances — {name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "feature_importance_dt.png"), dpi=120)
    plt.close(fig)
    print(f"[INFO] Feature importance plot saved.")


# ── Logistic Regression coefficients ──────────────────────────────────────────
def save_lr_coefficients(model):
    if not hasattr(model, "coef_"):
        return
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    coefs         = model.coef_[0]
    indices       = np.argsort(np.abs(coefs))[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if coefs[i] > 0 else "steelblue" for i in reversed(indices)]
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        [coefs[i]          for i in reversed(indices)],
        color=colors,
    )
    ax.set_title("Top-15 Feature Coefficients — Logistic Regression")
    ax.set_xlabel("Coefficient Value")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "feature_coef_lr.png"), dpi=120)
    plt.close(fig)
    print(f"[INFO] LR coefficient plot saved.")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Credit Risk — Training Pipeline")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    models  = get_models()
    metrics = {}

    for name, model in models.items():
        print(f"\n[INFO] Training: {name} ...")
        model.fit(X_train, y_train)

        # Save model
        safe_name = name.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        print(f"[INFO] Model saved → {model_path}")

        # Evaluate
        metrics[name] = evaluate(model, name, X_test, y_test)

    # Extra plots
    save_lr_coefficients(models["Logistic Regression"])
    save_feature_importance(models["Decision Tree"], "Decision Tree")

    # Persist metrics as JSON (used by Streamlit app)
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    # Convert confusion matrix (list of lists) to serialisable form
    metrics_to_save = {k: {kk: vv for kk, vv in v.items() if kk != "report"}
                       for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\n[INFO] Metrics saved → {metrics_path}")

    # ── Comparison table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    header = f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f}")

    print("\n[✓] Training complete.\n")


if __name__ == "__main__":
    main()

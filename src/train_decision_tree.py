"""
train_decision_tree.py
======================
Credit Risk Scoring System — Milestone 1 (Mid-Sem)
Trains and evaluates a Decision Tree classifier.
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
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
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
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
    print("  Model  : Decision Tree")
    print("─" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "decision_tree.pkl"))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"], ax=ax)
    ax.set_title("Confusion Matrix — Decision Tree")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "cm_decision_tree.png"), dpi=120)
    plt.close(fig)

    # Feature importance plot (top 15)
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    importances   = model.feature_importances_
    indices       = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in reversed(indices)],
            [importances[i]   for i in reversed(indices)],
            color="steelblue")
    ax.set_title("Top-15 Feature Importances — Decision Tree")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "feature_importance_dt.png"), dpi=120)
    plt.close(fig)

    return {
        "accuracy":         round(float(acc), 4),
        "roc_auc":          round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Decision Tree — Training")
    print("=" * 60)
    metrics = run()
    print(f"\n[✓] Done.  Accuracy={metrics['accuracy']}  ROC-AUC={metrics['roc_auc']}\n")

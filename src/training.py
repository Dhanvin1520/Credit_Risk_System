"""
training.py  (Orchestrator)
===========================
Credit Risk Scoring System — Milestone 1 (Mid-Sem)

Runs both model training scripts in sequence and saves a combined metrics.json.

You can also run each model independently:
    python src/train_logistic_regression.py
    python src/train_decision_tree.py
"""

import os
import json
import sys

# Make sure sibling modules are importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_logistic_regression
import train_decision_tree

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def main():
    print("=" * 60)
    print("  Credit Risk — Training Pipeline (Orchestrator)")
    print("=" * 60)

    metrics = {}

    print("\n[1/2] Training Logistic Regression ...")
    metrics["Logistic Regression"] = train_logistic_regression.run()

    print("\n[2/2] Training Decision Tree ...")
    metrics["Decision Tree"] = train_decision_tree.run()

    # Save combined metrics for the Streamlit app
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Combined metrics saved → {metrics_path}")

    # Comparison table
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

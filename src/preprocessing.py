"""
preprocessing.py
================
Credit Risk Scoring System — Milestone 1 (Mid-Sem)

Steps:
    1. Load raw CSV
    2. Remove / cap obvious outliers
    3. Encode categorical features
    4. Scale numeric features
    5. Stratified train/test split (80/20)
    6. Save processed arrays + scaler to models/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "Data", "loan_credit_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap obvious data-entry errors using domain knowledge."""
    before = len(df)

    # Age: realistic working-adult range
    df = df[df["person_age"].between(18, 80)]

    # Employment experience cannot exceed age - 16 (schooling years)
    df = df[df["person_emp_exp"] <= 50]

    # Income: cap at 99th percentile to remove extreme skew
    income_cap = df["person_income"].quantile(0.99)
    df = df[df["person_income"] <= income_cap]

    after = len(df)
    print(f"[INFO] Outlier removal: {before:,} → {after:,} rows "
          f"(removed {before - after:,})")
    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; binary columns → int."""

    # Binary column
    df["previous_loan_defaults_on_file"] = (
        df["previous_loan_defaults_on_file"].astype(int)
    )

    # One-hot encode nominal columns (drop_first avoids dummy trap)
    cat_cols = ["person_gender", "person_home_ownership", "loan_intent"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"[INFO] After encoding: {df.shape[1]} columns")
    return df


def scale_and_split(df: pd.DataFrame):
    """
    Scale features and produce an 80/20 stratified split.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    TARGET   = "loan_status"
    y        = df[TARGET].astype(int).values
    X        = df.drop(columns=[TARGET])

    feature_names = X.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ── Class distribution ─────────────────────────────────────────────────────
    unique, counts = np.unique(y_train, return_counts=True)
    print("[INFO] Training class distribution:")
    for cls, cnt in zip(unique, counts):
        label = "Default" if cls == 1 else "Non-Default"
        print(f"       Class {cls} ({label}): {cnt:,}  "
              f"({cnt/len(y_train)*100:.1f}%)")

    print(f"[INFO] Train: {X_train.shape},  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names):
    """Persist processed data and scaler for the training script."""
    np.save(os.path.join(MODEL_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(MODEL_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(MODEL_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"),  y_test)
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"[INFO] Artifacts saved to: {MODEL_DIR}/")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Credit Risk — Preprocessing Pipeline")
    print("=" * 60)

    df = load_data(DATA_PATH)
    df = remove_outliers(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = scale_and_split(df)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names)

    print("\n[] Preprocessing complete.\n")


if __name__ == "__main__":
    main()

"""
app.py
======
Credit Risk Scoring System — Streamlit UI (Milestone 1 / Mid-Sem)

Tabs:
    1. Single Borrower Prediction
    2. Batch CSV Upload
    3. Model Performance
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .risk-high   { background-color:#ffe0e0; border-left:6px solid #e53935;
                   padding:12px; border-radius:6px; }
    .risk-medium { background-color:#fff3e0; border-left:6px solid #fb8c00;
                   padding:12px; border-radius:6px; }
    .risk-low    { background-color:#e8f5e9; border-left:6px solid #43a047;
                   padding:12px; border-radius:6px; }
    .metric-box  { background:#fff; border-radius:10px; padding:16px;
                   text-align:center; box-shadow:0 2px 8px rgba(0,0,0,.08); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    lr_model      = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    dt_model      = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return scaler, feature_names, lr_model, dt_model, metrics


def build_input_df(row_dict, feature_names):
    """Convert user inputs into a correctly ordered & encoded DataFrame."""
    df = pd.DataFrame([row_dict])

    # One-hot encode without dropping first (reindex will handle the drops naturally)
    cat_cols = ["person_gender", "person_home_ownership", "loan_intent"]
    df = pd.get_dummies(df, columns=cat_cols)

    # Align to training feature order (drops reference dummy & fills missing with 0)
    df = df.reindex(columns=feature_names, fill_value=0)
    # Convert any boolean columns created by get_dummies to integers
    df = df.astype(float)
    return df


def risk_badge(prob: float):
    if prob >= 0.60:
        return "high",   " HIGH RISK"
    elif prob >= 0.35:
        return "medium", " MEDIUM RISK"
    else:
        return "low",    " LOW RISK"


# ── Load everything ─────────────────────────────────────────────────────────────
try:
    scaler, feature_names, lr_model, dt_model, metrics = load_artifacts()
    MODELS = {
        "Logistic Regression": lr_model,
        "Decision Tree":       dt_model,
    }
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    st.markdown("##  Credit Risk System")
    st.markdown("*Intelligent Lending Decision Support*")
    st.divider()

    if artifacts_ready:
        model_choice = st.selectbox(" Select Model", list(MODELS.keys()))
        st.divider()
        st.markdown("**About this system**")
        st.info(
            "This ML-powered tool predicts borrower creditworthiness "
            "using Logistic Regression and Decision Tree models trained "
            "on 45,000 historical loan records."
        )
    else:
        st.error(" Models not found. Please run:\n\n"
                 "```\npython src/preprocessing.py\npython src/training.py\n```")


# ── Tabs ────────────────────────────────────────────────────────────────────────
st.title(" Intelligent Credit Risk Scoring System")
st.markdown("*Milestone 1 — Machine Learning Based Credit Risk Prediction*")
st.divider()

if not artifacts_ready:
    st.error("Models not found. Run `preprocessing.py` then `training.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "  Single Borrower Prediction",
    "  Batch CSV Upload",
    "  Model Performance",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Borrower Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(" Single Borrower Risk Assessment")
    st.markdown("Fill in the borrower details below and click **Predict**.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("####  Personal Information")
        person_age    = st.slider("Age", 18, 80, 30)
        person_gender = st.selectbox("Gender", ["male", "female"])
        person_income = st.number_input(
            "Annual Income (₹ / $)", min_value=5000, max_value=500000,
            value=60000, step=1000)
        person_emp_exp = st.slider("Employment Experience (years)", 0, 50, 3)
        person_home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        person_education_encoded = st.selectbox(
            "Education Level",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "High School", 1: "Associate", 2: "Bachelor",
                3: "Master", 4: "Doctorate"
            }[x]
        )

    with c2:
        st.markdown("####  Loan Details")
        loan_amnt    = st.number_input(
            "Loan Amount (₹ / $)", min_value=500, max_value=35000,
            value=10000, step=500)
        loan_intent  = st.selectbox(
            "Loan Purpose",
            ["EDUCATION", "MEDICAL", "PERSONAL",
             "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_int_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 11.0, 0.01)
        loan_percent_income = round(loan_amnt / person_income, 4) if person_income else 0
        st.metric("Loan-to-Income Ratio", f"{loan_percent_income:.2%}")

    with c3:
        st.markdown("####  Credit History")
        cb_person_cred_hist_length = st.slider(
            "Credit History Length (years)", 2, 30, 5)
        credit_score = st.slider("Credit Score", 390, 850, 650)
        previous_loan_defaults_on_file = st.radio(
            "Previous Loan Default?", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True)

    st.divider()
    predict_btn = st.button(" Predict Credit Risk", type="primary", use_container_width=True)

    if predict_btn:
        raw = {
            "person_age":                      person_age,
            "person_gender":                   person_gender,
            "person_income":                   person_income,
            "person_emp_exp":                  person_emp_exp,
            "person_home_ownership":           person_home_ownership,
            "loan_amnt":                       loan_amnt,
            "loan_intent":                     loan_intent,
            "loan_int_rate":                   loan_int_rate,
            "loan_percent_income":             loan_percent_income,
            "cb_person_cred_hist_length":      cb_person_cred_hist_length,
            "credit_score":                    credit_score,
            "previous_loan_defaults_on_file":  previous_loan_defaults_on_file,
            "person_education_encoded":        person_education_encoded,
        }

        input_df  = build_input_df(raw, feature_names)
        scaled    = scaler.transform(input_df)
        model     = MODELS[model_choice]
        prob      = model.predict_proba(scaled)[0][1]
        risk_lvl, badge_text = risk_badge(prob)

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(f'<div class="metric-box"><h2>{prob*100:.1f}%</h2>'
                        f'<p>Default Probability</p></div>', unsafe_allow_html=True)
        with res_c2:
            st.markdown(f'<div class="metric-box"><h2>{(1-prob)*100:.1f}%</h2>'
                        f'<p>Creditworthiness Score</p></div>', unsafe_allow_html=True)
        with res_c3:
            st.markdown(f'<div class="metric-box"><h2>{badge_text}</h2>'
                        f'<p>Risk Classification</p></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="risk-{risk_lvl}" style="margin-top:16px">'
                    f'<b>{badge_text}</b> — Model: {model_choice} | '
                    f'Default probability: {prob*100:.2f}%</div>',
                    unsafe_allow_html=True)

        # Risk-driver table (LR coefficients × feature values)
        if model_choice == "Logistic Regression":
            coefs = model.coef_[0]
            contrib = pd.DataFrame({
                "Feature": feature_names,
                "Value":   scaled[0],
                "Coeff":   coefs,
            })
            contrib["Risk Contribution"] = contrib["Value"] * contrib["Coeff"]
            contrib = contrib.reindex(
                contrib["Risk Contribution"].abs().sort_values(ascending=False).index
            ).head(8)
            st.markdown("####  Top Risk Drivers")
            st.dataframe(contrib[["Feature", "Risk Contribution"]].reset_index(drop=True),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(" Batch Credit Risk Assessment")
    st.info("Upload a CSV with borrower data. "
            "The file should have the same columns as the training dataset.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write(f"**Loaded:** {len(batch_df):,} rows × {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(5), use_container_width=True)

            # Drop target if present
            if "loan_status" in batch_df.columns:
                true_labels = batch_df["loan_status"].values
                batch_df    = batch_df.drop(columns=["loan_status"])
            else:
                true_labels = None

            # Encode & scale
            cat_cols = ["person_gender", "person_home_ownership", "loan_intent"]
            batch_enc = pd.get_dummies(batch_df, columns=cat_cols)
            batch_enc = batch_enc.reindex(columns=feature_names, fill_value=0).astype(float)
            batch_scaled = scaler.transform(batch_enc)

            model = MODELS[model_choice]
            probs  = model.predict_proba(batch_scaled)[:, 1]
            preds  = model.predict(batch_scaled)

            result_df = batch_df.copy()
            result_df["Default_Probability_%"] = (probs * 100).round(2)
            result_df["Predicted_Default"]     = preds
            result_df["Risk_Level"]            = [
                risk_badge(p)[1] for p in probs
            ]

            if true_labels is not None:
                from sklearn.metrics import accuracy_score, roc_auc_score
                acc = accuracy_score(true_labels, preds)
                rac = roc_auc_score(true_labels, probs)
                m1, m2 = st.columns(2)
                m1.metric("Batch Accuracy",  f"{acc:.4f}")
                m2.metric("Batch ROC-AUC",   f"{rac:.4f}")

            st.dataframe(result_df, use_container_width=True)

            # Download
            csv_bytes = result_df.to_csv(index=False).encode()
            st.download_button(
                " Download Results as CSV",
                data=csv_bytes,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Distribution chart
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(probs, bins=30, color="steelblue", edgecolor="white")
            ax.set_xlabel("Default Probability")
            ax.set_ylabel("Number of Borrowers")
            ax.set_title("Distribution of Default Probabilities")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(" Model Performance Metrics")

    # ── Summary cards ───────────────────────────────────────────────────────────
    cols = st.columns(len(metrics))
    for col, (m_name, m_vals) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"**{m_name}**")
            st.metric("Accuracy",  f"{m_vals['accuracy']:.4f}")
            st.metric("ROC-AUC",   f"{m_vals['roc_auc']:.4f}")
            st.divider()

    # ── Confusion matrices ──────────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(2)
    cm_images = {
        "Logistic Regression": "cm_logistic_regression.png",
        "Decision Tree":       "cm_decision_tree.png",
    }
    for col, (title, fname) in zip(cm_cols, cm_images.items()):
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            col.image(fpath, caption=f"Confusion Matrix — {title}",
                      use_container_width=True)
        else:
            col.warning(f"Image not found: {fname}")

    # ── Feature importance / coefficients ───────────────────────────────────────
    st.markdown("### Feature Insights")
    fi_col1, fi_col2 = st.columns(2)
    fi_images = [
        ("feature_coef_lr.png",       "LR — Feature Coefficients"),
        ("feature_importance_dt.png", "Decision Tree — Feature Importances"),
    ]
    for col, (fname, caption) in zip([fi_col1, fi_col2], fi_images):
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            col.image(fpath, caption=caption, use_container_width=True)
        else:
            col.warning(f"Image not found: {fname}")

    # ── ROC Curve (requires test data) ─────────────────────────────────────────
    X_test_path = os.path.join(MODEL_DIR, "X_test.npy")
    y_test_path = os.path.join(MODEL_DIR, "y_test.npy")
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        st.markdown("### ROC Curves")
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"Logistic Regression": "royalblue", "Decision Tree": "tomato"}
        for m_name, m_obj in MODELS.items():
            y_prob = m_obj.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[m_name], lw=2,
                    label=f"{m_name} (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Model Comparison")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.caption(" Credit Risk Scoring System · Milestone 1 · Mid-Semester Submission")

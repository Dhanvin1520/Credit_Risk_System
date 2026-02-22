#  Intelligent Credit Risk Scoring System
### Google GenAI Capstone — Milestone 1 (Mid-Semester Submission)

---

##  Problem Statement
Predict borrower creditworthiness and default probability from historical loan data using supervised machine learning, without LLMs or Agentic AI.

---

##  Project Structure
```
Credit_Risk-_System/
├── Data/
│   └── loan_credit_data.csv       # 45,000 borrower records
├── src/
│   ├── preprocessing.py           # Data cleaning, encoding, scaling, splitting
│   ├── train_logistic_regression.py  # Standalone LR model training
│   ├── train_decision_tree.py     # Standalone DT model training
│   ├── training.py                # Orchestrator to run both models
│   └── app.py                     # Streamlit web application
├── models/                        # Auto-generated: saved models & plots
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Dataset Overview
| Property | Value |
|---|---|
| Rows | 45,000 |
| Features | 14 columns |
| Target | `loan_status` (0 = no default, 1 = default) |
| Missing values | None |
| Class distribution | 78% Non-Default / 22% Default |

---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Credit_Risk-_System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
python src/preprocessing.py
```
> Outputs: scaled train/test arrays + `scaler.pkl` in `models/`

### 4. Train Models
```bash
python src/training.py
```
> Outputs: `logistic_regression.pkl`, `decision_tree.pkl`, `metrics.json`, confusion matrix & feature plots in `models/`

### 5. Launch Streamlit App
```bash
streamlit run src/app.py
```
> Opens at `http://localhost:8501`

---

##  Models Used
| Model | Role | Key Parameters | Accuracy | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | Primary | `max_iter=1000` | **89.7%** | **0.953** |
| **Decision Tree** | Secondary | `max_depth=12`, `min_samples_leaf=15` | **92.3%** | **0.966** |

Both models are evaluated on **Accuracy**, **ROC-AUC**, **Classification Report**, and **Confusion Matrix**.

---

##  UI Features (Streamlit)
| Tab | Description |
|---|---|
|  Single Borrower | Input borrower details via sliders/dropdowns → get risk score, probability & badge |
|  Batch CSV Upload | Upload CSV → predict all rows → download results |
|  Model Performance | View accuracy, ROC-AUC, confusion matrices, feature plots |

---

##  Evaluation Metrics
- **Accuracy** — Overall correct predictions
- **ROC-AUC** — Ability to discriminate defaults from non-defaults
- **Confusion Matrix** — TP, FP, TN, FN breakdown
- **Classification Report** — Precision, Recall, F1-score per class

---

##  Key Design Decisions
- **Outlier removal**: age capped at 80, employment experience at 50 yrs, income at 99th percentile
- **No data leakage**: scaler fitted only on training set, applied to test set
- **Stratified split**: maintains class ratio across train and test
- **Prediction Consistency**: Kept class weights default so Logistic Regression maintains calibrated probability outputs compared to Decision Tree.

---

##  Team
Google GenAI Capstone Project — Milestone 1 (Mid-Semester)

---
* No LLMs, Agentic AI, or external APIs used in this milestone.*

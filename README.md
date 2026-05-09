# Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)
![MLflow](https://img.shields.io/badge/MLflow-2.8.0-green)

A machine learning classifier that predicts which customers are likely to churn, using only behavioral and subscription features. The model is deployed behind a **Streamlit dashboard** where users can upload a CSV and receive real‑time predictions.

## 📁 Data

- **Source:** Telco Customer Churn (Kaggle / IBM)
- **Size:** 7,032 customers after cleaning
- **Target:** `Churn` (Yes/No) – 26.6% churn rate (moderate imbalance)

## 🧹 Data Preparation & Feature Engineering

- Converted `TotalCharges` from string to numeric; dropped 11 blank rows.
- Encoded binary categorical columns (`Yes`/`No` → 1/0), and combined `No internet service` with `No` where appropriate.
- One‑hot encoded multi‑category columns (`Contract`, `InternetService`, `PaymentMethod`, `gender`).
- Filled missing `MultipleLines` for customers without phone service.
- No scaling required (tree‑based model).

## 🌲 Model & Tuning

| Experiment | Recall (Churn) | Precision (Churn) | F1 (Churn) | Description |
|------------|----------------|-------------------|------------|-------------|
| Baseline RF (class_weight='balanced') | 0.50 | 0.63 | 0.55 | Default threshold 0.5 |
| Tuned threshold (0.30) | **0.74** | **0.53** | **0.61** | Final deployed model |

**Why threshold tuning?**  
Because churn detection is imbalanced, we lowered the decision threshold from 0.5 → 0.30 to catch 74% of actual churners (recall), while still keeping false alarms around 50%. This is a common business trade‑off where missing a churner is costlier than offering a retention incentive.

**Final hyperparameters:**  
- `n_estimators`: 100  
- `class_weight`: 'balanced' (automatic)  
- `threshold`: 0.30 (post‑training)

## 📊 Feature Importance

Top predictors of churn:
- **TotalCharges**, **tenure**, **MonthlyCharges** – billing and loyalty dominate.
- **Contract type** (two‑year / one‑year) – longer contracts drastically reduce churn.
- **InternetService (Fiber optic)** – higher churn than DSL or no internet.
- **PaymentMethod (Electronic check)** – strong risk indicator.

This insight helps businesses design targeted retention campaigns (e.g., offer tech support and contract upgrades to high‑risk customers).

## 🚀 Deployment (Streamlit App)

The model is deployed as an interactive web app using **Streamlit**:

- Upload a CSV of customer data → get instant churn predictions and probabilities.
- Decision threshold is displayed and can be modified in the code.
- Download predictions as a CSV.

**How to run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Predictor")
st.write("Upload a CSV file with customer data. The model will predict who is at risk of churning.")

# Load model, threshold, and feature names
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read())
    with open("feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, threshold, feature_names

model, threshold, feature_names = load_artifacts()

st.markdown(f"✅ Model loaded. Decision threshold: **{threshold}**")
st.info("The CSV must contain the same columns as the training data (excluding the target 'Churn').")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Check required columns
        missing = [col for col in feature_names if col not in input_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Make predictions
            features = input_df[feature_names]
            probs = model.predict_proba(features)[:, 1]
            input_df["Churn Probability"] = probs
            # FIXED: use np.where instead of .map()
            input_df["Churn Prediction"] = np.where(probs >= threshold, "Yes", "No")

            st.subheader("Predictions")
            st.dataframe(input_df[["Churn Probability", "Churn Prediction"]])

            # Download button
            csv = input_df.to_csv(index=False)
            st.download_button("Download Full Results as CSV", csv, "churn_predictions.csv")

            # Summary
            churn_count = (input_df["Churn Prediction"] == "Yes").sum()
            st.metric("Customers Flagged as Churn Risk", churn_count)

    except Exception as e:
        st.error(f"An error occurred: {e}")

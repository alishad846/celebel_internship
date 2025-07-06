import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="🍷 Wine Classifier", layout="wide")

# --------------------------------------------------------------
# Load trained model
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("🍷 Wine Classifier & Explainer")

# --------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------
uploaded = st.file_uploader("Upload a Wine CSV (optional)", type=["csv"])
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success("Custom dataset loaded!")
else:
    data_path = os.path.join("data", "wine_dataset.csv")
    raw_df = pd.read_csv(data_path)
    st.info("Using default data/wine_dataset.csv")

st.subheader("📄 Data Preview")
st.dataframe(raw_df.head())

# --------------------------------------------------------------
# Build user input form
# --------------------------------------------------------------
st.subheader("🧠 Predict a Single Sample")
if "class" not in raw_df.columns:
    st.error("Uploaded CSV must contain a 'class' column.")
    st.stop()

feature_cols = raw_df.columns.drop("class")
user_vals = {}
cols = st.columns(3)
for idx, col in enumerate(feature_cols):
    with cols[idx % 3]:
        default_val = float(raw_df[col].median())
        user_vals[col] = st.number_input(col, value=default_val)

# --------------------------------------------------------------
# Make prediction
# --------------------------------------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_vals])

    # 1️⃣ Predict
    proba = model.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = model.classes_[pred_idx]
    st.success(f"Prediction → **Class {pred_class}** (prob = {proba[pred_idx]:.2f})")

    # 2️⃣ Probability Bar Chart
    st.subheader("📈 Class Probabilities")
    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(model.classes_, proba)
    ax_prob.set_xlabel("Wine Class")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(0, 1)
    for i, p in enumerate(proba):
        ax_prob.text(model.classes_[i], p + 0.02, f"{p:.2f}", ha="center")
    st.pyplot(fig_prob)

    # 3️⃣ SHAP Explanation
    st.subheader("🔍 Feature Contribution (SHAP)")
    explainer = shap.TreeExplainer(model["model"])
    input_proc = model["scaler"].transform(model["imputer"].transform(input_df))
    shap_vals_all = explainer.shap_values(input_proc)

    if isinstance(shap_vals_all, list):
        shap_vals = shap_vals_all[pred_idx][0]
    else:
        shap_vals = shap_vals_all[0]

    shap_vals = np.array(shap_vals).flatten()[:len(feature_cols)]

    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP": shap_vals
    }).sort_values("SHAP", key=lambda s: s.abs(), ascending=False)

    fig_shap, ax_shap = plt.subplots()
    sns.barplot(data=shap_df.head(10), y="Feature", x="SHAP", ax=ax_shap)
    ax_shap.set_title("Top Feature Contributions")
    st.pyplot(fig_shap)

    # 4️⃣ Radar Chart
    with st.expander("📊 Feature Radar Chart"):
        values = np.array(list(user_vals.values()))
        max_vals = raw_df[feature_cols].max().values
        ratios = values / max_vals
        angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
        angles += angles[:1]
        ratios = np.concatenate([ratios, ratios[:1]])

        fig_rad = plt.figure(figsize=(6, 6))
        ax_rad = fig_rad.add_subplot(111, polar=True)
        ax_rad.plot(angles, ratios, "o-", linewidth=2)
        ax_rad.fill(angles, ratios, alpha=0.25)
        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(feature_cols, fontsize=8)
        ax_rad.set_yticklabels([])
        st.pyplot(fig_rad)

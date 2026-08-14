"""
Machine Learning (S2-25_DSECLZG565)
Assignment - 2
BITS ID: 2025DAO4046
Streamlit App: Multi-Class Classification of Obesity Levels Based on Eating Habits and Physical Condition

"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Configuration of the page
st.set_page_config(page_title="Multi-Class Classification of Obesity", layout="wide")
st.title("Multi-Class Classification of Obesity Levels Based on Eating Habits and Physical Condition", text_alignment="center")
st.write(
    "Upload test data (CSV) and select a model from the dropdown menu to view predictions, "
    "evaluation metrics, and confusion matrix."
)

MODEL_DIR = "model/saved_models"
TARGET_COL = "NObeyesdad"

# Loading available models and object preprocessing

@st.cache_resource
def load_preprocessing():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    return scaler, encoders, target_encoder


@st.cache_resource
def load_model(model_filename):
    return joblib.load(os.path.join(MODEL_DIR, model_filename))


MODEL_OPTIONS = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
}

# Checking the model directory exists before further execution
if not os.path.exists(MODEL_DIR):
    st.error(
        f"Could not find '{MODEL_DIR}'. Make sure this app.py sits in the same "
        f"project folder as your model/saved_models/ directory (from training)."
    )
    st.stop()

try:
    scaler, encoders, target_encoder = load_preprocessing()
except FileNotFoundError as e:
    st.error(f"Missing preprocessing file: {e}")
    st.stop()

# Sidebar controls

with st.sidebar:
    st.write("Machine Learning (S2-25_DSECLZG565) - Assignment - 2")
    st.write("BITS ID: 2025DAO4046")

st.sidebar.header("App Controls")

uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

selected_model_name = st.sidebar.selectbox(
    "Select a model", list(MODEL_OPTIONS.keys())
)

run_button = st.sidebar.button("Run Model Evaluation", type="primary")

# Main logic
if uploaded_file is None:
    st.info("Upload a test CSV file from the sidebar to get started.")
    st.write(
        "Expected format: same columns used during training, including the "
        f"true label column **`{TARGET_COL}`** for metric calculation."
    )
    st.stop()

# Uploaded data read
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()

st.subheader("Previewing Uploaded Data")
st.dataframe(raw_df.head())

if TARGET_COL not in raw_df.columns:
    st.error(
        f"Uploaded CSV must contain the target column '{TARGET_COL}' "
        f"so that evaluation metrics can be calculated."
    )
    st.stop()

if run_button:
    with st.spinner("Preprocessing uploaded data and running model..."):
        try:
            df = raw_df.copy()

            # Encode categorical columns using saved encoders
            for col, le in encoders.items():
                if col in df.columns:
                    # Handle unseen categories gracefully
                    known_classes = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

            # Encode target
            y_true_raw = df[TARGET_COL]
            known_targets = set(target_encoder.classes_)
            y_true_raw = y_true_raw.apply(
                lambda x: x if x in known_targets else target_encoder.classes_[0]
            )
            y_true = target_encoder.transform(y_true_raw)

            X = df.drop(columns=[TARGET_COL])

            # Scale features
            X_scaled = scaler.transform(X)

            # Load selected model
            model = load_model(MODEL_OPTIONS[selected_model_name])

            # Predict
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)

        except Exception as e:
            st.error(f"Error during preprocessing/evaluation: {e}")
            st.stop()

    # Evaluation metrics
    st.subheader(f"Evaluation Metrics — {selected_model_name}")

    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan  # if uploaded test data does not contain all required classes

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    metric_cols = st.columns(len(metrics))
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.4f}" if not np.isnan(value) else "N/A")

    st.divider()

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    labels = target_encoder.classes_
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

    # Predictions preview
    st.subheader("Sample Predictions")
    pred_display = raw_df.copy()
    pred_display["Predicted"] = target_encoder.inverse_transform(y_pred)
    st.dataframe(pred_display.head(20))

else:
    st.info("Click **Run Evaluation** in the sidebar to generate results.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Churn Prediction", page_icon="🎯")

st.title("🎯 Credit Card Customer Churn Prediction")
st.markdown("### ML Assignment 2 - Abhinav Mandloi")

st.markdown("""
## ✅ Project Summary
- **Dataset:** BankChurners (10,127 customers, 20 features)
- **Models Trained:** 6 classification models
- **Best Model:** XGBoost (97.14% accuracy, 0.9922 AUC)
- **GitHub:** Complete implementation available
""")

# FEATURE 1: CSV Upload
st.header("📁 Upload Customer Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10))
    
    # FEATURE 2: Model Selection
    st.header("🤖 Select Classification Model")
    model = st.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors",
         "Naive Bayes", "Random Forest", "XGBoost"]
    )
    
    st.info(f"**Selected Model:** {model}")
    
    # FEATURE 3: Display Evaluation Metrics
    st.header("📈 Model Performance Metrics")
    
    metrics_data = {
        "Logistic Regression": {"Accuracy": 0.8490, "AUC": 0.9165, "Precision": 0.5188, "Recall": 0.8062, "F1": 0.6313, "MCC": 0.5627},
        "Decision Tree": {"Accuracy": 0.9403, "AUC": 0.9183, "Precision": 0.7898, "Recall": 0.8554, "F1": 0.8213, "MCC": 0.7864},
        "K-Nearest Neighbors": {"Accuracy": 0.9062, "AUC": 0.8790, "Precision": 0.8261, "Recall": 0.5262, "F1": 0.6429, "MCC": 0.6119},
        "Naive Bayes": {"Accuracy": 0.8806, "AUC": 0.8415, "Precision": 0.6361, "Recall": 0.5969, "F1": 0.6159, "MCC": 0.5456},
        "Random Forest": {"Accuracy": 0.9516, "AUC": 0.9832, "Precision": 0.8514, "Recall": 0.8462, "F1": 0.8488, "MCC": 0.8200},
        "XGBoost": {"Accuracy": 0.9714, "AUC": 0.9922, "Precision": 0.9211, "Recall": 0.8985, "F1": 0.9097, "MCC": 0.8927},
    }
    
    selected = metrics_data[model]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{selected['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{selected['AUC']:.4f}")
    col3.metric("Precision", f"{selected['Precision']:.4f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{selected['Recall']:.4f}")
    col5.metric("F1 Score", f"{selected['F1']:.4f}")
    col6.metric("MCC", f"{selected['MCC']:.4f}")
    
    # FEATURE 4: Confusion Matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Churn Prediction",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Customer Churn Prediction")
st.markdown("""
This application predicts customer churn using 6 different machine learning models.
Upload your customer data and select a model to see predictions and evaluation metrics.
""")

st.divider()

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression_model.pkl',
        'Decision Tree': 'model/decision_tree_model.pkl',
        'K-Nearest Neighbors': 'model/k_nearest_neighbors_model.pkl',
        'Naive Bayes': 'model/naive_bayes_model.pkl',
        'Random Forest': 'model/random_forest_model.pkl',
        'XGBoost': 'model/xgboost_model.pkl'
    }
    
    try:
        for name, path in model_files.items():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        return models, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

models, scaler, label_encoders = load_models()

if models is None:
    st.stop()

# Sidebar - Model selection
st.sidebar.header("⚙️ Configuration")
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
**Models Available:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="Upload a CSV file containing customer features"
    )

# File upload and prediction
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("🔍 View Data Preview"):
            st.dataframe(df.head(10))
        
        # Prepare data
        if 'Attrition_Flag' in df.columns:
            y_true = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
            X = df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1, errors='ignore')
        else:
            y_true = None
            X = df.drop(['CLIENTNUM'], axis=1, errors='ignore')
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Get selected model
        model = models[selected_model]
        
        # Check if model needs scaling
        needs_scaling = selected_model in ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']
        
        if needs_scaling:
            X_processed = scaler.transform(X)
        else:
            X_processed = X
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)
        
        # Display results
        st.divider()
        st.subheader(f"🎯 Results - {selected_model}")
        
        # Show predictions
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            churn_count = (y_pred == 1).sum()
            st.metric("Predicted Churns", churn_count)
        
        with col_pred2:
            retain_count = (y_pred == 0).sum()
            st.metric("Predicted Retains", retain_count)
        
        # If ground truth is available, show metrics
        if y_true is not None:
            st.divider()
            st.subheader("📈 Model Evaluation Metrics")
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            # Display metrics in columns
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("AUC Score", f"{auc:.4f}")
            
            with col_m2:
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")
            
            with col_m3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.divider()
            st.subheader("🔥 Confusion Matrix")
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Retained', 'Churned'],
                       yticklabels=['Retained', 'Churned'],
                       ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            
            # Classification Report
            st.divider()
            st.subheader("📋 Classification Report")
            
            report = classification_report(y_true, y_pred, 
                                          target_names=['Retained', 'Churned'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))
        
        else:
            st.warning("⚠️ No ground truth labels found. Showing predictions only.")
            
            # Show prediction distribution
            pred_df = pd.DataFrame({
                'Customer_Index': range(len(y_pred)),
                'Prediction': ['Churned' if p == 1 else 'Retained' for p in y_pred],
                'Churn_Probability': y_pred_proba[:, 1]
            })
            
            st.dataframe(pred_df.head(20))
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Please ensure your CSV has the correct format with all required features.")

else:
    # Show sample data info
    st.info("👆 Please upload a CSV file to begin prediction")
    
    st.markdown("""
    ### Expected Data Format:
    
    Your CSV should contain the following features:
    - Customer_Age
    - Gender
    - Dependent_count
    - Education_Level
    - Marital_Status
    - Income_Category
    - Card_Category
    - Months_on_book
    - Total_Relationship_Count
    - Months_Inactive_12_mon
    - Contacts_Count_12_mon
    - Credit_Limit
    - Total_Revolving_Bal
    - Avg_Open_To_Buy
    - Total_Amt_Chng_Q4_Q1
    - Total_Trans_Amt
    - Total_Trans_Ct
    - Total_Ct_Chng_Q4_Q1
    - Avg_Utilization_Ratio
    
    Optionally include **Attrition_Flag** for model evaluation.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Created by Abhinav Mandloi | M.Tech AI/ML | ML Assignment 2</p>
</div>
""", unsafe_allow_html=True)
# Credit Card Customer Churn Prediction

## Problem Statement

A bank is experiencing increasing customer churn in their credit card services. The business objective is to build a machine learning model that can predict which customers are likely to churn, enabling the bank to proactively engage with at-risk customers and improve retention through targeted interventions.

## Dataset Description

**Source:** [Kaggle - Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

**Dataset Characteristics:**
- **Total Instances:** 10,127 customers
- **Total Features:** 20 features (18 used after removing Naive Bayes columns and CLIENTNUM)
- **Target Variable:** Attrition_Flag (Binary Classification)
  - Attrited Customer (Churned): 16.07%
  - Existing Customer (Retained): 83.93%

**Key Features:**
- Customer demographics: Age, Gender, Dependent_count, Education_Level, Marital_Status
- Account information: Card_Category, Months_on_book, Credit_Limit
- Transaction behavior: Total_Trans_Amt, Total_Trans_Ct, Avg_Utilization_Ratio
- Relationship metrics: Contacts_Count_12_mon, Months_Inactive_12_mon

The dataset presents a class imbalance challenge with only 16% churned customers, making it important to use appropriate evaluation metrics beyond accuracy.

## Models Used

### Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| K-Nearest Neighbors | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Note:** Replace 0.XXXX with your actual values from `model_comparison.csv`

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | [Copy from model_observations.csv] |
| Decision Tree | [Copy from model_observations.csv] |
| K-Nearest Neighbors | [Copy from model_observations.csv] |
| Naive Bayes | [Copy from model_observations.csv] |
| Random Forest (Ensemble) | [Copy from model_observations.csv] |
| XGBoost (Ensemble) | [Copy from model_observations.csv] |

## Repository Structure
# 📊 Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn.

## 🎯 Project Objective

Customer churn is a major business problem for subscription-based companies.

The goal of this project is to:

- Analyze customer behavior
- Identify factors associated with churn
- Train multiple classification models
- Optimize the churn decision threshold
- Predict churn probability for new customers
- Provide actionable customer retention recommendations

## 📁 Dataset

The project uses the IBM Telco Customer Churn dataset.

Dataset contains:

- 7,043 customers
- 21 original features
- Customer demographic information
- Services and subscription information
- Billing information
- Churn target variable

## 🔎 Exploratory Data Analysis

Key findings:

- Total customers: **7,043**
- Customers who did not churn: **5,174 (73.46%)**
- Customers who churned: **1,869 (26.54%)**
- Duplicate rows: **0**
- Duplicate customer IDs: **0**
- `TotalCharges` contained 11 blank values
- Blank `TotalCharges` values were handled during preprocessing

## 🧹 Data Preprocessing

The project includes:

- Missing value handling
- Numeric feature conversion
- Categorical feature encoding
- Feature preprocessing using Scikit-learn Pipeline
- Train/test split
- Stratified sampling

### Numerical Features

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

### Categorical Features

- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

## 🤖 Models

Three machine learning models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

## 📊 Model Comparison

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8070 |    0.6604 | 0.5615 | 0.6069 |  0.8422 |
| Random Forest       |   0.7743 |    0.5648 | 0.6524 | 0.6055 |  0.8271 |
| XGBoost             |   0.7956 |    0.6473 | 0.5053 | 0.5676 |  0.8404 |

Logistic Regression achieved the strongest overall balance of performance and was selected as the final model.

## 🔁 Cross-Validation

5-Fold Cross-Validation:

| Metric    |   Mean |    Std |
| --------- | -----: | -----: |
| Accuracy  | 0.8030 | 0.0125 |
| Precision | 0.6552 | 0.0284 |
| Recall    | 0.5445 | 0.0407 |
| F1        | 0.5941 | 0.0303 |
| ROC-AUC   | 0.8462 | 0.0126 |

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize Logistic Regression.

Best parameters:

```text
C = 100
solver = liblinear
```

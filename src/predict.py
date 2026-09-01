import joblib
import pandas as pd


# Load trained pipeline
model = joblib.load("models/churn_logistic_pipeline.pkl")


# Example customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0
}])


# Prediction
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]


print("\n===== CUSTOMER CHURN PREDICTION =====")

print(f"Churn Probability: {probability:.2%}")

if prediction == 1:
    print("Prediction: HIGH CHURN RISK")
else:
    print("Prediction: LOW CHURN RISK")

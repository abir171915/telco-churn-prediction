import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load model and scaler once at startup ──
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Define the app ──
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a customer will churn based on their profile",
    version="1.0"
)

# ── Define what input data looks like ──
# This is Pydantic — it validates incoming data automatically
class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# ── Health check endpoint ──
@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running",
        "status": "healthy"
    }

# ── Prediction endpoint ──
@app.post("/predict")
def predict(customer: CustomerData):

    # Convert input to dataframe
    input_dict = customer.dict()
    input_df = pd.DataFrame([input_dict])

    # Rename columns to match training data
    input_df.columns = input_df.columns.str.replace('_', ' ')

    # Scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Align columns with model
    model_cols = model.feature_names_in_.tolist()
    input_df = input_df.reindex(columns=model_cols, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    churn_prob = round(float(probability[1]) * 100, 2)
    stay_prob  = round(float(probability[0]) * 100, 2)

    # Risk level
    if churn_prob >= 70:
        risk = "HIGH — call immediately with retention offer"
    elif churn_prob >= 40:
        risk = "MEDIUM — send personalised email offer"
    else:
        risk = "LOW — no action needed"

    return {
        "prediction": "WILL CHURN" if prediction == 1 else "WILL STAY",
        "churn_probability": f"{churn_prob}%",
        "stay_probability":  f"{stay_prob}%",
        "risk_level": risk
    }
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("random_forest_pipeline.pkl")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0"
)

# -----------------------------
# Input Schema (RAW DATA)
# -----------------------------
class CreditInput(BaseModel):
    age: int
    sex: str
    job: int
    housing: str
    saving_accounts: str
    checking_account: str
    credit_amount: float
    duration: int
    purpose: str
    credit_per_month: float
    age_risk_band: str          
    job_stability: int
    housing_stability: int
    has_saving_account: int
    has_checking_account: int


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"status": "Credit Risk API is running 🚀"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_credit_risk(data: CreditInput):

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # -----------------------------
        # FORCE SAFE DATA TYPES
        # -----------------------------
        numeric_cols = [
            "age", "job", "credit_amount", "duration",
            "credit_per_month", "job_stability",
            "housing_stability", "has_saving_account",
            "has_checking_account"
        ]

        categorical_cols = [
            "sex", "housing", "saving_accounts",
            "checking_account", "purpose",
            "age_risk_band"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # -----------------------------
        # EXACT COLUMN ORDER (CRITICAL)
        # -----------------------------
        df = df[
            [
                "age", "sex", "job", "housing",
                "saving_accounts", "checking_account",
                "credit_amount", "duration", "purpose",
                "credit_per_month", "age_risk_band",
                "job_stability", "housing_stability",
                "has_saving_account", "has_checking_account"
            ]
        ]

        # -----------------------------
        # Prediction
        # -----------------------------
        prob = model.predict_proba(df)[0][1]
        prediction = int(prob >= 0.45)

        return {
            "default_probability": round(float(prob), 3),
            "prediction": prediction,
            "risk_label": "High Risk" if prediction == 1 else "Low Risk"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

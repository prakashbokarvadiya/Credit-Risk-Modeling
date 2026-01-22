import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

model = joblib.load("random_forest_pipeline.pkl")

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def health():
    return {"status": "API Running"}

@app.post("/predict")
def predict(data: CreditInput):
    try:
        df = pd.DataFrame([data.dict()])
        prob = model.predict_proba(df)[0][1]
        return {
            "default_probability": round(float(prob), 3),
            "risk": "High" if prob >= 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

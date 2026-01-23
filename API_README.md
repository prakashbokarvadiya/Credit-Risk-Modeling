# Credit Risk Prediction API

## Overview
This API provides credit risk prediction for banking and financial use cases.
It predicts the probability that a customer will default on a loan and returns
a risk-based decision aligned with banking business logic.

The API is built using FastAPI and serves a trained Random Forest credit risk model
saved as a production-ready pipeline.

---

## Business Objective
Banks need to answer one critical question:

Which customers are likely to default on a loan?

This API helps:
- Identify high-risk customers
- Reduce loan default losses
- Support risk-aware loan approval decisions

---

## Model Details
- Model: Random Forest Classifier
- Training Approach: Cost-sensitive learning (class_weight)
- Evaluation Metrics:
  - ROC-AUC
  - Precision-Recall AUC
- Decision Strategy:
  - Probability-based prediction
  - Business-driven risk threshold

---

## API Endpoints

### Health Check
GET /

Response:
{
  "status": "Credit Risk API is running"
}

---

### Credit Risk Prediction
POST /predict

Predicts loan default risk for a single customer.

---

## Input Schema (Example)

{
  "age": 35,
  "sex": "male",
  "job": 2,
  "housing": "own",
  "saving_accounts": "little",
  "checking_account": "moderate",
  "credit_amount": 4500,
  "duration": 24,
  "purpose": "car",
  "credit_per_month": 187.5,
  "age_risk_band": "30-45",
  "job_stability": 1,
  "housing_stability": 1,
  "has_saving_account": 1,
  "has_checking_account": 1
}

---

## Output Response (Example)

{
  "default_probability": 0.62,
  "prediction": 1,
  "risk_label": "High Risk"
}

---

## Risk Interpretation
- Low Risk: Loan can be approved
- High Risk: Loan should be rejected or manually reviewed

The decision threshold is based on banking business costs where
missing a defaulter is more expensive than rejecting a good customer.

---

## Data Validation & Safety
- Strict input validation using Pydantic
- Forced numeric and categorical type casting
- Exact feature order enforcement
- Error handling using HTTP exceptions

---

## Run API Locally

1. Install dependencies
pip install fastapi uvicorn pandas scikit-learn xgboost joblib

2. Start server
uvicorn app:app --reload

3. Open Swagger UI
http://127.0.0.1:8000/docs

---

## Project Structure

app.py
credit_risk_random_forest.pkl
german_credit_cleaned.csv
README.md

---

## Key Takeaway
This API demonstrates how a machine learning model can be deployed
as a banking-grade service combining predictive accuracy with
business-driven decision logic.

---

## Author
Prakash Bokarvadiya

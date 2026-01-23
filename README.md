# Credit Risk Modeling (Banking Domain)

## Project Overview
This project focuses on Credit Risk Modeling in a banking context. 
The objective is to predict whether a customer is likely to default on a loan, 
helping banks make risk-aware loan approval decisions and minimize financial losses.

This is a real-world banking problem where incorrect predictions can lead to significant business impact.

---

## Problem Statement
Which customers are likely to default on a loan?

Banks need a system that can identify risky customers in advance so that loan default losses can be reduced.

---

## Dataset
- Dataset: German Credit Dataset (cleaned version)
- Total records: 1000
- Target variable: risk
  - 1 = Default
  - 0 = Non-default

Key features include age, job, housing, credit amount, loan duration, savings account status, and checking account status.

---

## Why This Project Is Banking-Level Hard
- Data is imbalanced (defaulters are fewer)
- False negative errors are very costly for banks
- Accuracy is not a reliable metric
- Business cost must drive model decisions

---

## Feature Engineering
- Credit per month = credit_amount / duration
- Job stability and housing stability indicators
- Savings and checking account flags

These features help capture customer financial stability and stress.

---

## Modeling Approach
Models used:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Random Forest was selected as the final model due to better Precision-Recall performance and stable results on imbalanced data.

---

## Evaluation Metrics
- ROC-AUC
- Precision-Recall AUC (primary focus)

Model Performance (approx):
- Random Forest: ROC-AUC ~0.78, PR-AUC ~0.66
- XGBoost: ROC-AUC ~0.76, PR-AUC ~0.62
- Logistic Regression: ROC-AUC ~0.74, PR-AUC ~0.59

---

## Cost-Sensitive Learning
In banking:
- False Negative (default approved) is very costly
- False Positive (good customer rejected) has lower cost

Business costs assumed:
- False Negative cost = 100,000
- False Positive cost = 10,000

Class-weighted models were used instead of SMOTE to keep the solution production-safe.

---

## Decision Threshold Optimization
Instead of using the default 0.5 threshold, multiple thresholds were tested using total business loss.

Final threshold selected: 0.35  
This makes the model risk-averse and reduces missed defaulters.

---

## Risk Bucket Strategy
Customers are categorized into:
- Low Risk (< 0.35): Approve
- Medium Risk (0.35 – 0.6): Manual Review
- High Risk (> 0.6): Reject

This reflects real banking decision processes.

---

## Model Deployment
- Final model saved using joblib
- Can be deployed via an API (e.g., FastAPI)
- Output includes default probability and risk category

---

## Key Takeaway
This project demonstrates how machine learning combined with business logic can be used to build a banking-grade credit risk model that focuses on minimizing real financial loss.

---

## Author
Prakash Bokarvadiya

# Transaction Fraud Risk Detection System

> End-to-end machine learning project for detecting fraudulent credit card transactions using Python, Scikit-learn, class imbalance handling, fraud risk scoring, threshold tuning, and business impact analysis.

---

## Project Overview

Financial institutions process millions of transactions every day, and fraudulent transactions are usually rare compared with legitimate transactions. This project builds a complete **Transaction Fraud Risk Detection System** that helps identify suspicious credit card transactions and prioritize them for review.

The project focuses on:

- Analyzing **284,807 credit card transactions** with a fraud rate of approximately **0.17%**
- Handling extreme class imbalance using techniques such as `class_weight="balanced"` and optional SMOTE
- Training and comparing machine learning models including Logistic Regression, Random Forest, Gradient Boosting, and optional XGBoost
- Evaluating models using fraud-focused metrics such as Recall, Precision, F1-Score, ROC-AUC, and PR-AUC
- Creating a **Fraud Risk Score from 0 to 100** for each transaction
- Performing threshold tuning to balance fraud detection and false alerts
- Estimating business impact using fraud loss prevention and manual review cost assumptions

---

## Business Problem

Traditional rule-based fraud systems can struggle to keep up with changing fraud patterns. Fraudulent transactions are uncommon, but even a small number of missed fraud cases can create financial loss, compliance risk, and customer trust issues.

This machine learning system is designed to help fraud and risk teams:

- Prioritize high-risk transactions for manual review
- Reduce false negatives, which represent missed fraud cases
- Control false positives, which increase analyst workload
- Support risk-based decision-making using adjustable probability thresholds
- Estimate the financial value of fraud detection improvements

---

## Dataset

**Credit Card Fraud Detection** — Kaggle dataset: `mlg-ulb/creditcardfraud`

| Feature | Description |
|---|---|
| `V1` – `V28` | Anonymized PCA-transformed transaction features |
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |
| `Amount` | Transaction amount |
| `Class` | Target variable: `1` = Fraud, `0` = Legitimate |

Dataset summary:

- **Total transactions:** 284,807
- **Fraud cases:** 492
- **Fraud rate:** Approximately 0.17%
- **Imbalance ratio:** Approximately 578 legitimate transactions for every 1 fraud transaction

---

## Tools and Technologies

| Category | Tools |
|---|---|
| Programming Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, optional XGBoost |
| Imbalance Handling | Class weights, optional imbalanced-learn / SMOTE |
| Environment | Google Colab |
| Version Control | Git, GitHub |

---

## Project Workflow

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning and Preprocessing
   ↓
Class Imbalance Handling
   ↓
Model Training and Comparison
   ↓
Model Evaluation
   ↓
Threshold Tuning
   ↓
Fraud Risk Scoring
   ↓
Business Impact Analysis
   
```

---

## Key Features

- **Flexible dataset loading:** Supports manual upload in Google Colab and optional Kaggle API download
- **Exploratory data analysis:** Class distribution, transaction amount analysis, time analysis, and correlation review
- **Imbalance-aware modeling:** Uses class weighting and optional SMOTE to improve fraud detection performance
- **Multiple model comparison:** Compares Logistic Regression, Random Forest, Gradient Boosting, optional XGBoost, and optional SMOTE-based model
- **Fraud-focused metrics:** Uses Recall, Precision, F1-Score, ROC-AUC, and PR-AUC instead of relying only on accuracy
- **Threshold tuning:** Compares probability thresholds to find the best trade-off between missed fraud and false alerts
- **Fraud risk scoring:** Converts model probabilities into a risk score from 0 to 100
- **Risk categorization:** Labels transactions as Low, Medium, or High Risk
- **Business impact simulation:** Estimates fraud loss prevented, manual review cost, and net financial impact
- **Export-ready outputs:** Generates CSV reports and summary files for portfolio presentation

---

## Machine Learning Models

| Model | Imbalance Strategy | Purpose |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | Interpretable baseline model |
| Random Forest | `class_weight="balanced"` | Ensemble model for nonlinear patterns |
| Gradient Boosting | Threshold tuning and evaluation on imbalanced data | Boosting model for stronger predictive performance |
| XGBoost | Optional `scale_pos_weight` | Advanced boosting model if available |
| Random Forest + SMOTE | Optional synthetic oversampling | Compares oversampling against class-weighted models |

---

## Evaluation Metrics

Fraud detection requires more than accuracy because the dataset is highly imbalanced. A model can achieve very high accuracy by predicting almost every transaction as legitimate, while still missing actual fraud.

| Metric | Why It Matters for Fraud Detection |
|---|---|
| **Recall** | Measures how many actual fraud cases were detected. Important for reducing missed fraud. |
| **Precision** | Measures how many flagged transactions were truly fraud. Important for reducing false alerts. |
| **F1-Score** | Balances precision and recall into one metric. |
| **ROC-AUC** | Measures the model’s ability to separate fraud and legitimate transactions. |
| **PR-AUC** | Especially useful for imbalanced datasets because it focuses on fraud-class performance. |

---

## Results Summary

The notebook compares model performance using fraud-focused metrics and selects the best model based on recall, precision-recall balance, PR-AUC, and estimated business impact.

Example model comparison output:

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook |
| Random Forest | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook |
| Gradient Boosting | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook |
| XGBoost | Optional | Optional | Optional | Optional | Optional |

The final model is selected after comparing:

- Fraud recall
- Precision-recall balance
- PR-AUC
- Threshold performance
- Estimated business impact

---

## Fraud Risk Scoring

The selected model generates a fraud probability for each transaction. That probability is converted into a **Fraud Risk Score**:

```text
Fraud Risk Score = Predicted Fraud Probability × 100
```

Risk categories:

| Risk Score | Risk Category |
|---:|---|
| 0–30 | Low Risk |
| 31–70 | Medium Risk |
| 71–100 | High Risk |

This makes the model output easier to understand for fraud operations teams and business stakeholders.

---

## Threshold Tuning

Instead of relying only on the default 0.50 probability threshold, this project tests multiple thresholds such as:

```text
0.10, 0.20, 0.30, 0.40, 0.50
```

For each threshold, the notebook calculates:

- Precision
- Recall
- F1-Score
- False positives
- False negatives
- Estimated financial impact

This helps select a practical fraud monitoring threshold based on business risk and operational review capacity.

---

## Estimated Business Impact

The project includes a business impact simulation using example assumptions:

| Assumption | Value |
|---|---:|
| Average fraud loss per missed fraud transaction | $150 |
| Manual review cost per flagged transaction | $5 |

The notebook calculates:

- Number of fraud cases detected
- Number of fraud cases missed
- Estimated fraud loss prevented
- Estimated manual review cost
- Estimated net savings

These assumptions are used for portfolio demonstration and can be adjusted based on business context.

---

## Visualizations

The notebook creates professional charts, including:

- Fraud vs legitimate transaction distribution
- Transaction amount distribution
- Transaction amount comparison by fraud class
- Correlation heatmap
- Model performance comparison
- Confusion matrix for selected model
- ROC curve
- Precision-recall curve
- Threshold tuning comparison
- Fraud risk category distribution
- Business impact comparison
- Feature importance, where available

---

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebook/transaction_fraud_risk_detection.ipynb`
3. Download `creditcard.csv` from Kaggle dataset `mlg-ulb/creditcardfraud`
4. Run the manual upload cell in the notebook and upload `creditcard.csv`
5. Run all notebook cells from top to bottom
6. Review charts, model metrics, fraud risk scores, and exported output files

## Future Enhancements

- Build a real-time fraud scoring API using FastAPI or Flask
- Add streaming transaction monitoring with Apache Kafka or AWS Kinesis
- Use Autoencodes for anomaly detection
- Add model monitoring for data drift and performance decay
- Add SHAP-based explainability for individual transactions
- Deploy the model using AWS SageMaker
- Build a Tableau or Power BI dashboard for fraud operations monitoring
- Add champion/challenger testing for fraud model comparison



## Author

**Rasesh Ravula**  
Data Analyst | Fraud & Risk Analytics | Python | SQL | Scikit-learn | Tableau | Power BI



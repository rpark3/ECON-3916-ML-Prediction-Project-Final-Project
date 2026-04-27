# ECON 3916 Final Project  
## Predicting Diabetes Risk from Health and Lifestyle Indicators

## Project Overview
This project builds a machine learning model to predict whether an individual is at higher risk of diabetes using health, demographic, and behavioral indicators from the CDC Diabetes Health Indicators dataset. The project is framed as a **prediction** problem, not a causal inference problem. The goal is to support early risk screening, not diagnose diabetes.

## Prediction Question
Can health, demographic, and behavioral indicators predict whether an individual has diabetes?

## Stakeholder
The main stakeholder is a **public health screening program, clinic, or preventive care organization** that wants to identify individuals who may benefit from additional screening, follow-up, or preventive education.

## Decision This Enables
This model can help stakeholders:
- identify higher-risk individuals
- prioritize follow-up screening
- target preventive outreach
- support education and early intervention

## Dataset
This project uses the **CDC Diabetes Health Indicators** dataset derived from **BRFSS 2015**.

- **Target variable:** `Diabetes_binary`
- **Observations:** 253,680
- **Predictor variables:** 21

### Example Predictors
- BMI
- General health
- Physical health
- Mental health
- High blood pressure
- High cholesterol
- Difficulty walking
- Smoking
- Physical activity
- Fruit and vegetable consumption
- Sex
- Age
- Education
- Income

## Final Model
Two models were compared:
1. Logistic Regression
2. Random Forest Classifier

The final model selected was **Logistic Regression (balanced)**.

### Why Logistic Regression Was Chosen
Although Random Forest achieved a very slightly higher ROC-AUC, the difference was negligible. Logistic Regression was selected because:
- it had **higher recall for the diabetes class**
- it is **more interpretable**
- it is **easier to deploy**
- it produces a **smaller saved model artifact**
- it better fits the goal of a **screening support tool**, where catching more potentially high-risk individuals matters

## Repo Contents
This repository includes:

- `3916_final_project_starter(1).ipynb` — final notebook with EDA, modeling, and evaluation
- `app (2).py` — Streamlit dashboard
- `model.pkl` — saved trained model used by the app
- `requirements.txt` — Python dependencies
- `ECON 3916_ ML Prediction Project Report (4).pdf` — final 5-page report
- `AI Appendix` — AI methodology appendix
- `Code Link` — code-related submission material
- `Streamlit App Link` — deployed app link

## Reproducibility Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ECON-3916-ML-Prediction-Project-Final-Project

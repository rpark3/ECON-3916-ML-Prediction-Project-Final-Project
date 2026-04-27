# Predicting Diabetes Risk from Health and Lifestyle Indicators

## Project Overview
This project builds a machine learning model to predict whether an individual is at higher risk of diabetes using health, demographic, and behavioral indicators. The goal is not to diagnose diabetes or make causal claims, but to support early risk screening and help stakeholders identify individuals who may benefit from additional follow-up, education, or preventive care.

## Prediction Question
Can health, demographic, and behavioral indicators predict whether an individual has diabetes?

## Stakeholder
The main stakeholder is a public health screening program, clinic, or preventive care organization that needs a practical way to identify individuals at elevated diabetes risk.

## Decision This Enables
The model can help stakeholders prioritize:
- follow-up screening
- preventive outreach
- patient education
- earlier evaluation for higher-risk individuals

## Dataset
**Name:** CDC Diabetes Health Indicators Dataset  
**Source:** BRFSS 2015  
**Target Variable:** `Diabetes_binary`  
**Observations:** 253,680  
**Features:** 21 predictor variables

### Example Predictors
- BMI
- General health
- Physical health
- Mental health
- High blood pressure
- High cholesterol
- Difficulty walking
- Physical activity
- Smoking
- Fruit and vegetable consumption
- Sex
- Age
- Education
- Income

## Data Quality Summary
- The cleaned dataset contained **no missing values**
- The main data quality issue was **class imbalance**
- Potential outliers were examined using **IQR / Tukey-fence logic**
- Outliers were retained unless clearly implausible, since extreme values may reflect real high-risk individuals

## Methodology
This project was framed as a **binary classification** problem.

### Models Compared
1. **Logistic Regression**
2. **Random Forest Classifier**

### Train/Test Split
- 80% training
- 20% test
- `random_state = 42`
- stratified split used to preserve target balance

### Validation Approach
- 5-fold cross-validation
- primary comparison metric: **ROC-AUC**

## Results
Cross-validation showed that **Logistic Regression outperformed Random Forest** on ROC-AUC:

- **Model 1 (Logistic Regression):** 0.8227 +/- 0.0026
- **Model 2 (Random Forest):** 0.8015 +/- 0.0017

On the test set, Logistic Regression also produced the stronger ROC curve overall.

### Important Limitation
Although Logistic Regression achieved strong ROC-AUC and high overall accuracy, its recall for the diabetes class was low. This means it missed many true higher-risk individuals, which limits its usefulness as a standalone screening model unless thresholding or class-balance strategies are improved.

## Key Takeaways
- The dataset contains meaningful predictive signal
- Logistic Regression performed better than Random Forest in this project
- Variables such as general health, high blood pressure, BMI, difficulty walking, high cholesterol, and age appeared strongly associated with diabetes status in EDA
- These findings are **associational**, not causal

## Streamlit App
This project also includes a Streamlit app that allows users to enter health and lifestyle information and receive:
- predicted diabetes risk
- model confidence
- an updated probability chart

The app is designed as a **screening support tool**, not a medical diagnosis tool.

## Files
- `app.py` — Streamlit application
- `model.pkl` — trained machine learning model
- `requirements.txt` — required Python packages
- `notebook.ipynb` — analysis and modeling workflow

## How to Run the App
1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt

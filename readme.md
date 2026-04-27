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

- Identify higher-risk individuals
- Prioritize follow-up screening
- Target preventive outreach
- Support education and early intervention

## Dataset

This project uses the **CDC Diabetes Health Indicators** dataset derived from **BRFSS 2015**.

- **Source:** CDC Behavioral Risk Factor Surveillance System / Diabetes Health Indicators dataset
- **Target variable:** `Diabetes_binary`
- **Observations:** 253,680
- **Predictor variables:** 21
- **Access date:** April 2026

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

## Prediction vs. Causation

This project is focused on **prediction**, not causal inference. The model estimates whether a person is likely to be classified as diabetic based on observed health and lifestyle indicators. It does **not** prove that any individual feature causes diabetes.

For example, if high blood pressure or BMI is important in the model, that means those variables are useful for prediction in this dataset. It does not mean the model has proven a causal relationship.

## Final Model

Two models were compared:

1. Logistic Regression
2. Random Forest Classifier

The final model selected was **balanced Logistic Regression**.

### Why Logistic Regression Was Chosen

Although Random Forest achieved a very slightly higher ROC-AUC, the difference was negligible. Logistic Regression was selected because:

- It had higher recall for the diabetes class
- It is more interpretable
- It is easier to deploy
- It produces a smaller saved model artifact
- It better fits the goal of a screening support tool, where catching more potentially high-risk individuals matters

## Methods Summary

The project uses:

- Exploratory data analysis
- Missing data and data quality checks
- Train/test split with `random_state=42`
- Logistic Regression with class balancing
- Random Forest Classifier comparison
- Cross-validation
- ROC-AUC, recall, precision, and classification metrics
- Feature importance interpreted as predictive importance, not causal effect
- Streamlit deployment for interactive prediction

## Repository Contents

This repository includes:

- `final_project_notebook.ipynb` — final notebook with EDA, modeling, and evaluation
- `app.py` — Streamlit dashboard
- `model.pkl` — saved trained model used by the app
- `requirements.txt` — Python dependencies
- `final_report.pdf` — final 5-page report
- `ai_methodology_appendix.pdf` — AI methodology appendix
- `README.md` — project overview and reproducibility instructions

## Reproducibility Instructions

Follow the steps below to reproduce the project locally.

### 1. Clone the Repository

Command:

git clone https://github.com/rpark3/ECON-3916-ML-Prediction-Project-Final-Project.git

Then enter the project folder:

cd ECON-3916-ML-Prediction-Project-Final-Project

### 2. Create a Virtual Environment

Command:

python -m venv venv

### 3. Activate the Virtual Environment

On Mac/Linux, run:

source venv/bin/activate

On Windows, run:

venv\Scripts\activate

### 4. Install Dependencies

Command:

pip install -r requirements.txt

### 5. Run the Jupyter Notebook

Command:

jupyter notebook final_project_notebook.ipynb

Run the notebook from top to bottom to reproduce the exploratory data analysis, model training, model comparison, evaluation metrics, and saved model artifact.

### 6. Launch the Streamlit App Locally

Command:

streamlit run app.py

The Streamlit app allows users to adjust health and lifestyle inputs and view the model’s predicted diabetes risk.

## Streamlit Dashboard

Deployed app: [Insert your Streamlit app link here]

## Results Summary

The Logistic Regression and Random Forest models both showed useful predictive performance. Random Forest performed slightly better by ROC-AUC, but the difference was small. Balanced Logistic Regression was selected as the final model because it performed well, was more interpretable, and was better suited for a screening-support setting.

Because this is a public health screening context, recall for the diabetes class was especially important. A model that identifies more potentially high-risk individuals is more useful for early outreach than a model that only maximizes overall accuracy.

## Recommendation

The model can be used as a screening-support tool to help identify individuals who may benefit from additional diabetes screening, preventive education, or follow-up care. It should be used to support decision-making, not replace clinical judgment.

The most appropriate use case is early risk flagging in a public health or preventive care setting.

## Important Limitation

This model is intended for **screening support only**. It should not be used as a diagnostic tool. The model identifies predictive patterns in the dataset, but these relationships should not be interpreted as causal effects.

The model also depends on the quality and representativeness of the BRFSS 2015 data. Predictions may not generalize perfectly to all populations, healthcare settings, or future years.

## AI Methodology

AI tools were used as a co-pilot during the project and documented using the P.R.I.M.E. framework:

- **Prep:** Defined the task and project goals before prompting
- **Request:** Wrote a specific prompt for help with coding, interpretation, or writing
- **Iterate:** Refined AI output when it was incomplete or unclear
- **Mechanism Check:** Verified that code worked and that outputs made sense
- **Evaluate:** Applied human judgment to decide what to keep, revise, or reject

The full documentation is included in the AI methodology appendix.

## Files Required to Reproduce

To reproduce the project, the following files should be present in the repository:

- `final_project_notebook.ipynb`
- `app.py`
- `model.pkl`
- `requirements.txt`
- `final_report.pdf`
- `ai_methodology_appendix.pdf`

## Notes for Users

This dashboard is designed for educational and analytical purposes. It should not be used to diagnose diabetes or make medical decisions. A high-risk prediction should be interpreted as a signal that further screening or clinical follow-up may be useful.

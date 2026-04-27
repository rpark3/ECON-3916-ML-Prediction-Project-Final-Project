# ECON 3916 Final Project — Predicting Diabetes Risk from Health and Lifestyle Indicators

## Project Overview
This project builds a machine learning model to predict whether an individual is at higher risk of diabetes using health, demographic, and behavioral indicators. The project is framed as a prediction problem, not a causal inference problem.

## Prediction Question
Can health, demographic, and behavioral indicators predict whether an individual has diabetes?

## Stakeholder
The main stakeholder is a public health screening program, clinic, or preventive care organization that wants to identify individuals who may benefit from additional screening, follow-up, or preventive education.

## Project Contents
- `report/ECON_3916_ML_Prediction_Project_Report.pdf` — final 5-page report
- `notebook/ECON_3916_final_project.ipynb` — full notebook with EDA, modeling, and evaluation
- `app/app.py` — Streamlit dashboard
- `app/model.pkl` — trained model used by the app
- `app/requirements.txt` — packages needed to run the app
- `appendix/AI_Methodology_Appendix.pdf` — documented AI interactions using the P.R.I.M.E. framework
- `data/data_instructions.txt` — dataset information and placement instructions

## Dataset
This project uses the CDC Diabetes Health Indicators dataset derived from BRFSS 2015.

Target variable:
- `Diabetes_binary`

Main predictors include:
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

## Reproducibility Instructions

### 1. Environment Setup
Create and activate a virtual environment if desired.

#### Mac/Linux
```bash
python -m venv venv
source venv/bin/activate

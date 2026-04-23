import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

st.title("Diabetes Risk Screener")
st.markdown(
    "This app predicts diabetes risk based on health, demographic, and behavioral indicators. "
    "It is intended for screening support only and is not a medical diagnosis."
)

# -----------------------------
# User inputs
# -----------------------------
HighBP = st.selectbox("High Blood Pressure", [0, 1])
HighChol = st.selectbox("High Cholesterol", [0, 1])
CholCheck = st.selectbox("Cholesterol Check in Last 5 Years", [0, 1])
BMI = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
Smoker = st.selectbox("Smoker", [0, 1])
Stroke = st.selectbox("Stroke History", [0, 1])
HeartDiseaseorAttack = st.selectbox("Heart Disease or Heart Attack", [0, 1])
PhysActivity = st.selectbox("Physical Activity", [0, 1])
Fruits = st.selectbox("Consumes Fruit Regularly", [0, 1])
Veggies = st.selectbox("Consumes Vegetables Regularly", [0, 1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])
AnyHealthcare = st.selectbox("Any Healthcare Coverage", [0, 1])
NoDocbcCost = st.selectbox("Could Not See Doctor Because of Cost", [0, 1])
GenHlth = st.selectbox("General Health (1 = excellent, 5 = poor)", [1, 2, 3, 4, 5])
MentHlth = st.slider("Poor Mental Health Days (past 30 days)", 0, 30, 0)
PhysHlth = st.slider("Poor Physical Health Days (past 30 days)", 0, 30, 0)
DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
Sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
Age = st.selectbox("Age Category", list(range(1, 14)))
Education = st.selectbox("Education Level", [1, 2, 3, 4, 5, 6])
Income = st.selectbox("Income Category", [1, 2, 3, 4, 5, 6, 7, 8])

# -----------------------------
# Build input dataframe
# -----------------------------
input_df = pd.DataFrame([{
    "HighBP": HighBP,
    "HighChol": HighChol,
    "CholCheck": CholCheck,
    "BMI": BMI,
    "Smoker": Smoker,
    "Stroke": Stroke,
    "HeartDiseaseorAttack": HeartDiseaseorAttack,
    "PhysActivity": PhysActivity,
    "Fruits": Fruits,
    "Veggies": Veggies,
    "HvyAlcoholConsump": HvyAlcoholConsump,
    "AnyHealthcare": AnyHealthcare,
    "NoDocbcCost": NoDocbcCost,
    "GenHlth": GenHlth,
    "MentHlth": MentHlth,
    "PhysHlth": PhysHlth,
    "DiffWalk": DiffWalk,
    "Sex": Sex,
    "Age": Age,
    "Education": Education,
    "Income": Income
}])

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]
prob_no_diabetes = probs[0]
prob_diabetes = probs[1]

st.subheader("Prediction Result")
st.write(f"**Predicted probability of diabetes:** {prob_diabetes:.2%}")
st.write(f"**Predicted probability of no diabetes:** {prob_no_diabetes:.2%}")

if prediction == 1:
    st.error("Model prediction: Higher diabetes risk")
else:
    st.success("Model prediction: Lower diabetes risk")

# -----------------------------
# Uncertainty / confidence
# -----------------------------
st.subheader("Model Uncertainty")
st.write(
    "Because this is a classification model, uncertainty is shown using predicted class probabilities "
    "rather than prediction intervals."
)

confidence = max(prob_diabetes, prob_no_diabetes)
st.write(f"**Model confidence in predicted class:** {confidence:.2%}")

if confidence >= 0.80:
    st.write("Confidence level: High")
elif confidence >= 0.65:
    st.write("Confidence level: Moderate")
else:
    st.write("Confidence level: Low")

# -----------------------------
# Live-updating chart
# -----------------------------
st.subheader("Risk Probability Chart")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["No Diabetes", "Diabetes"], [prob_no_diabetes, prob_diabetes])
ax.set_ylabel("Predicted Probability")
ax.set_ylim(0, 1)
ax.set_title("Predicted Class Probabilities")

st.pyplot(fig)

# Optional: show raw inputs
with st.expander("Show input values"):
    st.dataframe(input_df)

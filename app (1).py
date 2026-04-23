import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = joblib.load("model.pkl")  # change filename if yours is different

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Screener", layout="centered")

st.title("Diabetes Risk Screener")
st.markdown(
    """
This app estimates **diabetes risk** using health, demographic, and lifestyle information.

It is a **screening support tool**, not a medical diagnosis.
"""
)

st.markdown("---")

# --------------------------------------------------
# Mappings from user-friendly labels to model codes
# --------------------------------------------------
yes_no = {"No": 0, "Yes": 1}

sex_map = {
    "Female": 0,
    "Male": 1
}

genhlth_map = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}

age_map = {
    "18–24": 1,
    "25–29": 2,
    "30–34": 3,
    "35–39": 4,
    "40–44": 5,
    "45–49": 6,
    "50–54": 7,
    "55–59": 8,
    "60–64": 9,
    "65–69": 10,
    "70–74": 11,
    "75–79": 12,
    "80 or older": 13
}

education_map = {
    "Never attended school or kindergarten only": 1,
    "Grades 1 through 8": 2,
    "Grades 9 through 11": 3,
    "Grade 12 or GED": 4,
    "College 1 to 3 years": 5,
    "College 4 years or more": 6
}

income_map = {
    "Less than $10,000": 1,
    "$10,000 to less than $15,000": 2,
    "$15,000 to less than $20,000": 3,
    "$20,000 to less than $25,000": 4,
    "$25,000 to less than $35,000": 5,
    "$35,000 to less than $50,000": 6,
    "$50,000 to less than $75,000": 7,
    "$75,000 or more": 8
}

# --------------------------------------------------
# Input section
# --------------------------------------------------
st.header("Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    highbp_label = st.selectbox("Do you have high blood pressure?", list(yes_no.keys()))
    highchol_label = st.selectbox("Do you have high cholesterol?", list(yes_no.keys()))
    cholcheck_label = st.selectbox("Have you had a cholesterol check in the last 5 years?", list(yes_no.keys()))
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    smoker_label = st.selectbox("Have you smoked at least 100 cigarettes in your life?", list(yes_no.keys()))
    stroke_label = st.selectbox("Have you ever had a stroke?", list(yes_no.keys()))
    heart_label = st.selectbox("Have you ever had coronary heart disease or a heart attack?", list(yes_no.keys()))
    activity_label = st.selectbox("Did you do physical activity or exercise in the past 30 days?", list(yes_no.keys()))
    fruits_label = st.selectbox("Do you eat fruit 1 or more times per day?", list(yes_no.keys()))
    veggies_label = st.selectbox("Do you eat vegetables 1 or more times per day?", list(yes_no.keys()))
    alcohol_label = st.selectbox("Are you a heavy alcohol consumer?", list(yes_no.keys()))

with col2:
    healthcare_label = st.selectbox("Do you have any kind of healthcare coverage?", list(yes_no.keys()))
    nodoc_label = st.selectbox("Was there a time in the past 12 months when you needed a doctor but could not see one because of cost?", list(yes_no.keys()))
    genhlth_label = st.selectbox("How would you rate your general health?", list(genhlth_map.keys()))
    menthlth = st.slider("How many days in the past 30 days was your mental health not good?", 0, 30, 0)
    physhlth = st.slider("How many days in the past 30 days was your physical health not good?", 0, 30, 0)
    diffwalk_label = st.selectbox("Do you have serious difficulty walking or climbing stairs?", list(yes_no.keys()))
    sex_label = st.selectbox("Sex", list(sex_map.keys()))
    age_label = st.selectbox("Age", list(age_map.keys()))
    education_label = st.selectbox("Education level", list(education_map.keys()))
    income_label = st.selectbox("Income level", list(income_map.keys()))

# --------------------------------------------------
# Convert labels to model codes
# --------------------------------------------------
HighBP = yes_no[highbp_label]
HighChol = yes_no[highchol_label]
CholCheck = yes_no[cholcheck_label]
BMI = bmi
Smoker = yes_no[smoker_label]
Stroke = yes_no[stroke_label]
HeartDiseaseorAttack = yes_no[heart_label]
PhysActivity = yes_no[activity_label]
Fruits = yes_no[fruits_label]
Veggies = yes_no[veggies_label]
HvyAlcoholConsump = yes_no[alcohol_label]
AnyHealthcare = yes_no[healthcare_label]
NoDocbcCost = yes_no[nodoc_label]
GenHlth = genhlth_map[genhlth_label]
MentHlth = menthlth
PhysHlth = physhlth
DiffWalk = yes_no[diffwalk_label]
Sex = sex_map[sex_label]
Age = age_map[age_label]
Education = education_map[education_label]
Income = income_map[income_label]

# --------------------------------------------------
# Build model input
# --------------------------------------------------
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

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_no_diabetes = probabilities[0]
    prob_diabetes = probabilities[1]
    confidence = max(prob_no_diabetes, prob_diabetes)

    st.markdown("---")
    st.header("Prediction Result")

    st.write(f"**Predicted probability of diabetes:** {prob_diabetes:.2%}")
    st.write(f"**Predicted probability of no diabetes:** {prob_no_diabetes:.2%}")

    if prediction == 1:
        st.error("Prediction: Higher diabetes risk")
    else:
        st.success("Prediction: Lower diabetes risk")

    st.subheader("What the prediction means")
    st.markdown(
        """
- **Higher diabetes risk** means the model thinks your profile looks more similar to people in the dataset who were in the diabetes class.
- **Lower diabetes risk** means the model thinks your profile looks more similar to people in the dataset who were in the non-diabetes class.
- This is **not** a diagnosis.
"""
    )

    st.subheader("Model uncertainty")
    st.write(f"**Model confidence in this prediction:** {confidence:.2%}")

    if confidence >= 0.80:
        st.write("Confidence level: High")
    elif confidence >= 0.65:
        st.write("Confidence level: Moderate")
    else:
        st.write("Confidence level: Low")

    # Live-updating chart
    st.subheader("Risk Probability Chart")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No Diabetes", "Diabetes"], [prob_no_diabetes, prob_diabetes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Predicted Class Probabilities")
    st.pyplot(fig)

    with st.expander("Show the encoded input values sent to the model"):
        st.dataframe(input_df)

# --------------------------------------------------
# Variable explanations
# --------------------------------------------------
st.markdown("---")
st.header("Explanation of Inputs")

st.markdown(
    """
**High Blood Pressure / High Cholesterol / Cholesterol Check / Smoker / Stroke / Heart Disease / Physical Activity / Fruits / Vegetables / Heavy Alcohol Consumption / Healthcare Coverage / Could Not See Doctor Because of Cost / Difficulty Walking**
- These are all **Yes/No** questions.

**BMI**
- Body Mass Index, a weight-for-height measure.

**General Health**
- Excellent, Very good, Good, Fair, or Poor.

**Mental Health**
- Number of days in the last 30 days when mental health was not good.

**Physical Health**
- Number of days in the last 30 days when physical health was not good.

**Sex**
- Female or Male.

**Age**
- Age group category.

**Education**
- Highest level of education completed.

**Income**
- Household income category.
"""
)

st.caption("Built from the CDC Diabetes Health Indicators dataset.")

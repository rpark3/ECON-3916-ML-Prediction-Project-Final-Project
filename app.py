import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    layout="wide"
)

# Change this if your saved model file has a different name
MODEL_PATH = "model.pkl"

# Use your chosen classification threshold here
# If you did not finalize threshold tuning, keep 0.50 for now
THRESHOLD = 0.50

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# LABEL MAPS
# --------------------------------------------------
yes_no_map = {
    "No": 0,
    "Yes": 1
}

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
# TITLE / DESCRIPTION
# --------------------------------------------------
st.title("Diabetes Risk Prediction Dashboard")
st.markdown(
    """
This dashboard estimates **diabetes risk** using health, demographic, and behavioral indicators.

It is intended as a **screening support tool**, not a medical diagnosis tool.
"""
)

st.info(
    "Important: This model provides a predictive risk estimate based on survey-style inputs. "
    "It should support awareness and follow-up decisions, not replace professional medical judgment."
)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.sidebar.header("Enter Patient / User Information")

highbp_label = st.sidebar.selectbox(
    "High blood pressure",
    options=list(yes_no_map.keys()),
    help="Has the individual ever been told they have high blood pressure?"
)

highchol_label = st.sidebar.selectbox(
    "High cholesterol",
    options=list(yes_no_map.keys())
)

cholcheck_label = st.sidebar.selectbox(
    "Cholesterol check in the last 5 years",
    options=list(yes_no_map.keys())
)

bmi = st.sidebar.number_input(
    "Body Mass Index (BMI)",
    min_value=10.0,
    max_value=100.0,
    value=25.0,
    step=0.1
)

smoker_label = st.sidebar.selectbox(
    "Smoked at least 100 cigarettes in lifetime",
    options=list(yes_no_map.keys())
)

stroke_label = st.sidebar.selectbox(
    "History of stroke",
    options=list(yes_no_map.keys())
)

heart_label = st.sidebar.selectbox(
    "History of coronary heart disease or heart attack",
    options=list(yes_no_map.keys())
)

activity_label = st.sidebar.selectbox(
    "Physical activity in the past 30 days",
    options=list(yes_no_map.keys())
)

fruits_label = st.sidebar.selectbox(
    "Consumes fruit 1 or more times per day",
    options=list(yes_no_map.keys())
)

veggies_label = st.sidebar.selectbox(
    "Consumes vegetables 1 or more times per day",
    options=list(yes_no_map.keys())
)

alcohol_label = st.sidebar.selectbox(
    "Heavy alcohol consumption",
    options=list(yes_no_map.keys())
)

healthcare_label = st.sidebar.selectbox(
    "Any healthcare coverage",
    options=list(yes_no_map.keys())
)

nodoc_label = st.sidebar.selectbox(
    "Could not see doctor because of cost in past 12 months",
    options=list(yes_no_map.keys())
)

genhlth_label = st.sidebar.selectbox(
    "General health",
    options=list(genhlth_map.keys())
)

menthlth = st.sidebar.slider(
    "Days mental health was not good in past 30 days",
    min_value=0,
    max_value=30,
    value=0
)

physhlth = st.sidebar.slider(
    "Days physical health was not good in past 30 days",
    min_value=0,
    max_value=30,
    value=0
)

diffwalk_label = st.sidebar.selectbox(
    "Serious difficulty walking or climbing stairs",
    options=list(yes_no_map.keys())
)

sex_label = st.sidebar.selectbox(
    "Sex",
    options=list(sex_map.keys())
)

age_label = st.sidebar.selectbox(
    "Age group",
    options=list(age_map.keys())
)

education_label = st.sidebar.selectbox(
    "Education level",
    options=list(education_map.keys())
)

income_label = st.sidebar.selectbox(
    "Income level",
    options=list(income_map.keys())
)

# --------------------------------------------------
# ENCODE INPUTS
# --------------------------------------------------
input_df = pd.DataFrame([{
    "HighBP": yes_no_map[highbp_label],
    "HighChol": yes_no_map[highchol_label],
    "CholCheck": yes_no_map[cholcheck_label],
    "BMI": bmi,
    "Smoker": yes_no_map[smoker_label],
    "Stroke": yes_no_map[stroke_label],
    "HeartDiseaseorAttack": yes_no_map[heart_label],
    "PhysActivity": yes_no_map[activity_label],
    "Fruits": yes_no_map[fruits_label],
    "Veggies": yes_no_map[veggies_label],
    "HvyAlcoholConsump": yes_no_map[alcohol_label],
    "AnyHealthcare": yes_no_map[healthcare_label],
    "NoDocbcCost": yes_no_map[nodoc_label],
    "GenHlth": genhlth_map[genhlth_label],
    "MentHlth": menthlth,
    "PhysHlth": physhlth,
    "DiffWalk": yes_no_map[diffwalk_label],
    "Sex": sex_map[sex_label],
    "Age": age_map[age_label],
    "Education": education_map[education_label],
    "Income": income_map[income_label]
}])

# --------------------------------------------------
# PREDICTION (AUTO-UPDATING)
# --------------------------------------------------
probabilities = model.predict_proba(input_df)[0]
prob_no_diabetes = float(probabilities[0])
prob_diabetes = float(probabilities[1])

prediction = int(prob_diabetes >= THRESHOLD)
confidence = max(prob_no_diabetes, prob_diabetes)
distance_from_threshold = abs(prob_diabetes - THRESHOLD)

# --------------------------------------------------
# MAIN OUTPUT
# --------------------------------------------------
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("Prediction Output")

    st.metric(
        label="Predicted Probability of Diabetes",
        value=f"{prob_diabetes:.2%}"
    )

    st.metric(
        label="Predicted Probability of No Diabetes",
        value=f"{prob_no_diabetes:.2%}"
    )

    st.write(f"**Decision threshold:** {THRESHOLD:.2f}")

    if prediction == 1:
        st.error("Prediction: Higher diabetes risk")
    else:
        st.success("Prediction: Lower diabetes risk")

    st.subheader("Uncertainty")
    st.write(f"**Model confidence in predicted class:** {confidence:.2%}")

    if distance_from_threshold < 0.05:
        st.warning(
            "This prediction is close to the classification threshold, so it should be treated as less certain."
        )
    elif confidence >= 0.80:
        st.info("Confidence level: High")
    elif confidence >= 0.65:
        st.info("Confidence level: Moderate")
    else:
        st.info("Confidence level: Low")

    st.subheader("Interpretation")
    st.markdown(
        """
- A **higher diabetes risk** prediction means the input profile looks more similar to observations in the dataset classified as diabetes.
- A **lower diabetes risk** prediction means the input profile looks more similar to observations classified as non-diabetes.
- This is **predictive output**, not a diagnosis and not evidence of causation.
"""
    )

with right_col:
    st.subheader("Interactive Visualization")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["No Diabetes", "Diabetes"],
        [prob_no_diabetes, prob_diabetes]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Predicted Class Probabilities")
    st.pyplot(fig)

# --------------------------------------------------
# OPTIONAL INPUT SUMMARY
# --------------------------------------------------
with st.expander("Show encoded model inputs"):
    st.dataframe(input_df)

# --------------------------------------------------
# INPUT EXPLANATIONS
# --------------------------------------------------
st.markdown("---")
st.header("Input Definitions")

st.markdown(
    """
**High blood pressure / High cholesterol / Cholesterol check / Smoker / Stroke / Heart disease / Physical activity / Fruit intake / Vegetable intake / Heavy alcohol consumption / Healthcare coverage / Could not see doctor because of cost / Difficulty walking**
- These are all yes/no indicators.

**BMI**
- Body Mass Index, a standard weight-for-height measure.

**General health**
- Self-rated overall health: Excellent, Very good, Good, Fair, or Poor.

**Mental health**
- Number of days in the past 30 days when mental health was not good.

**Physical health**
- Number of days in the past 30 days when physical health was not good.

**Sex**
- Female or Male.

**Age group**
- Encoded age bracket used by the dataset.

**Education level**
- Highest level of education completed.

**Income level**
- Household income category used in the dataset.
"""
)

st.caption("Built using the CDC Diabetes Health Indicators dataset.")

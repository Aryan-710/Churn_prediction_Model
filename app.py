import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    with open("telco_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Preprocessing function
def preprocess_input(gender, senior_citizen, partner, dependents, tenure, monthly_charges):
    # Example of encoding
    data = {
        'gender_Male': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }

    # Add missing dummy variables if needed
    df = pd.DataFrame([data])
    # If training used more dummy columns, add them with default 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    return df[model.feature_names_in_]

# Streamlit UI
st.title("📞 Telco Customer Churn Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

if st.button("Predict"):
    try:
        input_df = preprocess_input(gender, senior_citizen, partner, dependents, tenure, monthly_charges)
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is NOT likely to churn.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

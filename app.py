import streamlit as st
import pickle
import numpy as np

# ------- Load Pre-trained Model --------
@st.cache_resource
def load_model():
    with open("telco_model.pkl", "rb") as file:  # Make sure to save model as pickle
        model = pickle.load(file)
    return model

model = load_model()

# ------- App Title -------
st.title("📞 Telco Customer Churn Prediction App")
st.write("Predict if a customer is likely to churn based on their details.")

# ------- User Input Form -------
st.header("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

# ------- Predict Button -------
if st.button("Predict Churn"):
    # Convert inputs to numerical values
    gender_val = 1 if gender == "Male" else 0
    senior_val = 1 if senior_citizen == "Yes" else 0
    partner_val = 1 if partner == "Yes" else 0
    dependents_val = 1 if dependents == "Yes" else 0

    # Combine inputs into array
    input_data = np.array([[gender_val, senior_val, partner_val, dependents_val, tenure, monthly_charges]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is NOT likely to churn.")

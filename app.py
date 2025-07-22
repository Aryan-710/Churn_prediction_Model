import streamlit as st
import pickle
import pandas as pd

# Load your trained ML model
@st.cache_resource
def load_model():
    with open("telco_model.pkl", "rb") as file:  # Replace with your actual model file
        model = pickle.load(file)
    return model

model = load_model()

# Hardcode feature names used during training
feature_columns = [
    'gender_Female', 'gender_Male', 'SeniorCitizen',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
    'tenure', 'MonthlyCharges',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Function to preprocess user inputs
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)

    # Add any missing columns (set them to 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[feature_columns]

    return df

# Streamlit UI
st.title("📞 Telco Customer Churn Prediction App")
st.write("Enter customer details to predict if they are likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Prediction button
if st.button("Predict"):
    user_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment_method
    }

    try:
        processed_input = preprocess_input(user_input)
        prediction = model.predict(processed_input)[0]

        if prediction == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is NOT likely to churn.")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

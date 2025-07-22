import streamlit as st
import pickle
import pandas as pd

# Load trained model
@st.cache_resource
def load_model():
    with open("telco_model (2).pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Feature columns used during training
feature_columns = [
    'gender_Female', 'gender_Male', 'SeniorCitizen',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
    'tenure', 'MonthlyCharges',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Preprocess user input
def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

# UI
st.title("📞 Telco Customer Churn Prediction App")
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Predict"):
    inputs = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": charges,
        "Contract": contract,
        "PaymentMethod": payment
    }
    try:
        processed = preprocess_input(inputs)
        pred = model.predict(processed)[0]
        st.success("✅ Customer will NOT churn." if pred == 0 else "⚠️ Customer WILL churn.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

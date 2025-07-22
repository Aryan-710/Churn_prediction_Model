import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("telco_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_columns"]

model, feature_columns = load_model()

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder exactly
    df = df[feature_columns]
    return df

# ==== STREAMLIT UI ====
st.title("📞 Telco Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Predict"):
    ui_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": charges,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaymentMethod": payment
    }

    try:
        X = preprocess_input(ui_data)
        y = model.predict(X)[0]
        if y == 0:
            st.success("✅ Customer will NOT churn.")
        else:
            st.warning("⚠️ Customer WILL churn.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

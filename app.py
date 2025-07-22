import streamlit as st
import pickle
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open("telco_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load feature columns used during training
@st.cache_resource
def load_feature_columns():
    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)
    return feature_columns

feature_columns = load_feature_columns()

# Preprocess user input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)  # One-hot encode user input

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order
    df = df[feature_columns]

    return df

# Streamlit UI
st.title("📞 Telco Customer Churn Prediction App")
st.write("Fill the details below to predict if the customer is likely to churn.")

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

# Predict button
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

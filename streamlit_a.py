import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer information to predict churn likelihood")

st.subheader("Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner = st.selectbox("Partner", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24)
    phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    multiple_lines = st.selectbox("Multiple Lines", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1569.0)
    gender_male = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

with col3:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

st.subheader("Streaming and Contract")
col4, col5, col6 = st.columns(3)

with col4:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col5:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"]
    )

with col6:
    st.write(" ")
    st.write(" ")

if st.button("Predict Churn"):
    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0

    online_security_no = 1 if online_security == "No internet service" else 0
    online_security_yes = 1 if online_security == "Yes" else 0

    online_backup_no = 1 if online_backup == "No internet service" else 0
    online_backup_yes = 1 if online_backup == "Yes" else 0

    device_protection_no = 1 if device_protection == "No internet service" else 0
    device_protection_yes = 1 if device_protection == "Yes" else 0

    tech_support_no = 1 if tech_support == "No internet service" else 0
    tech_support_yes = 1 if tech_support == "Yes" else 0

    streaming_tv_no = 1 if streaming_tv == "No internet service" else 0
    streaming_tv_yes = 1 if streaming_tv == "Yes" else 0

    streaming_movies_no = 1 if streaming_movies == "No internet service" else 0
    streaming_movies_yes = 1 if streaming_movies == "Yes" else 0

    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0

    payment_cc = 1 if payment_method == "Credit card (automatic)" else 0
    payment_ec = 1 if payment_method == "Electronic check" else 0
    payment_mc = 1 if payment_method == "Mailed check" else 0

    features = np.array([[
        senior_citizen,
        partner,
        dependents,
        tenure,
        phone_service,
        multiple_lines,
        paperless_billing,
        monthly_charges,
        total_charges,
        gender_male,
        internet_fiber,
        internet_no,
        online_security_no,
        online_security_yes,
        online_backup_no,
        online_backup_yes,
        device_protection_no,
        device_protection_yes,
        tech_support_no,
        tech_support_yes,
        streaming_tv_no,
        streaming_tv_yes,
        streaming_movies_no,
        streaming_movies_yes,
        contract_one_year,
        contract_two_year,
        payment_cc,
        payment_ec,
        payment_mc
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"Customer will churn (Probability: {probability:.2%})")
    else:
        st.success(f"Customer will stay (Churn Probability: {probability:.2%})")

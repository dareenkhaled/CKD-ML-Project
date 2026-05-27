
import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("kidney_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 Kidney Disease Prediction App")

# Input form
age = st.number_input("Age", 0, 120)
bp = st.number_input("Blood Pressure", 0, 200)
sg = st.number_input("Specific Gravity", 1.0, 1.030, step=0.001)
al = st.number_input("Albumin", 0, 5)
su = st.number_input("Sugar", 0, 5)
bgr = st.number_input("Blood Glucose Random", 0, 500)
bu = st.number_input("Blood Urea", 0, 200)
sc = st.number_input("Serum Creatinine", 0.0, 15.0, step=0.1)

if st.button("Predict"):
    user_data = np.array([[age, bp, sg, al, su, bgr, bu, sc]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of kidney disease!")
    else:
        st.success("✅ Low risk of kidney disease.")

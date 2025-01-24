import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained SVM model and scaler
model = joblib.load('svm_Model.pkl')
scaler = joblib.load('scaler.pkl')  # Make sure you saved the scaler

# Streamlit UI
st.title("Customer Purchase Prediction")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=100, value=25)
estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=150000, value=50000)

# Process input data
gender_encoded = 1 if gender == 'Female' else 0
input_data_unscaled = np.array([[age, estimated_salary]])  # Remove Gender
input_data_scaled = scaler.transform(input_data_unscaled)  # Scale numeric features
gender_encoded_reshaped = np.array([[gender_encoded]])  # Convert gender to 2D
input_data_final = np.hstack((gender_encoded_reshaped, input_data_scaled))  # Combine all features

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data_final)
    result = "Purchased" if prediction[0] == 1 else "Not Purchased"
    st.write(f"### Prediction: {result}")

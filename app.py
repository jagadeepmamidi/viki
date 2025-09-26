import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('disease_rf_model.pkl')

# Title
st.title('Disease Diagnosis Predictor')

# Sidebar inputs (edit features as per your dataset)
st.sidebar.header("Patient Data")
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
heart_rate = st.sidebar.number_input('Heart Rate (bpm)', min_value=40, max_value=200, value=80)
body_temp = st.sidebar.number_input('Body Temperature (C)', min_value=34.0, max_value=42.0, value=37.0)
blood_pressure = st.sidebar.number_input('Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
oxygen = st.sidebar.number_input('Oxygen Saturation (%)', min_value=85, max_value=100, value=98)
symptom1 = st.sidebar.text_input('Symptom 1')
symptom2 = st.sidebar.text_input('Symptom 2')
symptom3 = st.sidebar.text_input('Symptom 3')
severity = st.sidebar.selectbox('Severity', ['Mild', 'Moderate', 'Severe'])
treatment_plan = st.sidebar.selectbox('Treatment Plan', ['Rest and fluids', 'Medication and rest', 'Hospitalization and medication'])

# NOTE: You must encode categorical features and scale numerics exactly like you did in preprocessing!
# For demo purpose, we assume everything is already numerical or encoded.
input_dict = {
    'Age': age,
    'Gender': 1 if gender == 'Male' else 0,
    'HeartRatebpm': heart_rate,
    'BodyTemperatureC': body_temp,
    'BloodPressuremmHg': blood_pressure,  # Should be numerical or use appropriate encoding
    'OxygenSaturation': oxygen,
    'Symptom1': 0,  # Should use same encoding as training (update this logic for real deploy)
    'Symptom2': 0,
    'Symptom3': 0,
    'Severity': {'Mild':0, 'Moderate':1, 'Severe':2}[severity],
    'TreatmentPlan': {'Rest and fluids':0, 'Medication and rest':1, 'Hospitalization and medication':2}[treatment_plan],
    'Respiratory_Symptom': int('shortness of breath' in [symptom1, symptom2, symptom3] or 'cough' in [symptom1, symptom2, symptom3])
}

input_df = pd.DataFrame([input_dict])

if st.button('Predict Diagnosis'):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Diagnosis: {pred}")

st.write("**Tip:** For deployment, ensure all encoding/scaling matches your training pipeline exactly!")

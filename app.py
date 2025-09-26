import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('disease_rf_model.pkl')

SYMPTOM_CHOICES = [
    'Fatigue', 'Sore throat', 'Fever', 'Cough', 'Body ache',
    'Shortness of breath', 'Headache', 'Runny nose'
]

st.title('Disease Diagnosis Predictor')

st.sidebar.header("Patient Data")
age = st.sidebar.number_input('Age', min_value=18, max_value=80, value=30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
heart_rate = st.sidebar.number_input('Heart Rate (bpm)', min_value=60, max_value=120, value=80)
body_temp = st.sidebar.number_input('Body Temperature (C)', min_value=35.0, max_value=40.0, value=37.0)
blood_pressure = st.sidebar.number_input('Blood Pressure (mmHg)', min_value=80, max_value=160, value=120)
oxygen = st.sidebar.number_input('Oxygen Saturation (%)', min_value=90, max_value=99, value=98)
symptom1 = st.sidebar.selectbox('Symptom 1', SYMPTOM_CHOICES)
symptom2 = st.sidebar.selectbox('Symptom 2', SYMPTOM_CHOICES)
symptom3 = st.sidebar.selectbox('Symptom 3', SYMPTOM_CHOICES)
severity = st.sidebar.selectbox('Severity', ['Mild', 'Moderate', 'Severe'])
treatment_plan = st.sidebar.selectbox('Treatment Plan', ['Rest and fluids', 'Medication and rest', 'Hospitalization and medication'])

# Use EXACT feature names from training with underscores
input_dict = {
    'Patient_ID': 1,  # or any default integer (unless used in model meaningfully)
    'Age': age,
    'Gender': 1 if gender == 'Male' else 0,
    'Heart_Rate_bpm': heart_rate,
    'Body_Temperature_C': body_temp,
    'Blood_Pressure_mmHg': blood_pressure,
    'Oxygen_Saturation_%': oxygen,
    'Symptom_1': SYMPTOM_CHOICES.index(symptom1),
    'Symptom_2': SYMPTOM_CHOICES.index(symptom2),
    'Symptom_3': SYMPTOM_CHOICES.index(symptom3),
    'Severity': {'Mild':0, 'Moderate':1, 'Severe':2}[severity],
    'Treatment_Plan': {'Rest and fluids':0, 'Medication and rest':1, 'Hospitalization and medication':2}[treatment_plan],
    'Respiratory_Symptom': int('shortness of breath' in [symptom1, symptom2, symptom3] or 'cough' in [symptom1, symptom2, symptom3])
}
input_df = pd.DataFrame([input_dict])

if st.button('Predict Diagnosis'):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Diagnosis: {pred}")

st.write("""
**Important:**  
All feature names and categorical encodings MUST match those used during training, with underscores and exact casing as per your model!
""")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('student_grade_predictor.pkl')

# App title
st.title("🎓 Student Grade Predictor App")

st.write("Enter the student's details below:")

# Input features from user
studytime = st.slider('Study Time (1=low, 4=high)', 1, 4, 2)
absences = st.number_input('Number of Absences', min_value=0, max_value=100, value=0)
Dalc = st.slider('Workday Alcohol Consumption (1=very low, 5=very high)', 1, 5, 1)
Walc = st.slider('Weekend Alcohol Consumption (1=very low, 5=very high)', 1, 5, 1)
failures = st.slider('Number of Past Class Failures', 0, 3, 0)

# (Add more inputs if needed)

# When user clicks "Predict"
if st.button('Predict Final Grade'):
    # Create input dataframe
    input_data = {
        'studytime': [studytime],
        'absences': [absences],
        'Dalc': [Dalc],
        'Walc': [Walc],
        'failures': [failures],
    }
    input_df = pd.DataFrame(input_data)

    # Predict using model
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Final Grade (G3): {prediction:.2f}")

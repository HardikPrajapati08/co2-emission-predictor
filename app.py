# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model 
model = joblib.load("model.pkl") 

st.set_page_config(page_title="CO₂ Emission Predictor", layout="centered")
st.title("🚗 CO₂ Emission Predictor")
st.markdown("Predict CO₂ emissions based on vehicle features")

# Input features
cylinders = st.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
fuel_consumption = st.number_input("Fuel Consumption (L/100 km)", min_value=1.0, max_value=30.0, value=8.0, step=0.1)

# Predict button
if st.button("Predict CO₂ Emission"):
    input_data = pd.DataFrame({
        'Cylinders': [cylinders],
        'Engine_Size': [engine_size],
        'Fuel_Consumption': [fuel_consumption],
    })

    prediction = model.predict(input_data)
    st.success(f"Predicted CO₂ Emission: {prediction[0]:.2f} g/km")

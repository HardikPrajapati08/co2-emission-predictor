import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="CO₂ Emission Predictor", layout="centered")
st.title("🚗 CO₂ Emission Predictor")
st.markdown("Predict CO₂ emissions based on vehicle features")

# Input fields
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=1.2, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, value=4, step=1)
fuel_consumption = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=1.0, max_value=30.0, value=5.5, step=0.1)

if st.button("Predict CO₂ Emission"):
    # Use the exact column names used in training
    input_data = pd.DataFrame([{
        'ENGINESIZE': engine_size,
        'CYLINDERS': cylinders,
        'FUELCONSUMPTION_COMB': fuel_consumption
    }])

    # Predict
    prediction = model.predict(input_data)
    st.success(f"Predicted CO₂ Emission: {prediction[0]:.2f} g/km")

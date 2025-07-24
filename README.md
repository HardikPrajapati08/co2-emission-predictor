# 🚗 CO₂ Emission Prediction Project

Predicts vehicle CO₂ emissions using machine learning.

## Features

- Trains 3 models:
  - Linear Regression
  - Polynomial Regression
  - Random Forest Regressor
- Evaluates and compares their performance
- Saves the best model (Random Forest)
- Provides a beautiful Gradio GUI to run predictions

## Files

- `model_training.py` – Trains and saves the model
- `app.py` – GUI app using Gradio
- `FuelConsumptionCo2.csv` – Dataset
- `model.pkl` – Saved model
- `requirements.txt` – Python dependencies

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

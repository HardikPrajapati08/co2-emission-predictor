# model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("FuelConsumptionCo2.csv")

# Show initial data
print("📄 First 5 rows:")
print(data.head())

print("\n🧾 Columns:")
print(data.columns)

# Select features and target
features = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
target = data['CO2EMISSIONS']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 1. Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_lin_pred = lin_model.predict(X_test)

print("\n📈 Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_lin_pred))
print("R² Score:", r2_score(y_test, y_lin_pred))
print("Coefficients:")
for i, col in enumerate(features.columns):
    print(f"  {col}: {lin_model.coef_[i]}")
print("Intercept:", lin_model.intercept_)

# 2. Polynomial Regression (Degree 2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)

print("\n🔁 Polynomial Regression:")
print("R² Score:", r2_score(y_test, y_poly_pred))

# 3. Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

print("\n🌲 Random Forest:")
print("R² Score:", r2_score(y_test, y_rf_pred))

# Save Random Forest model
joblib.dump(rf_model, "model.pkl")
print("\n✅ Random Forest model saved as 'model.pkl'")

# Optional: Visualize prediction
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_rf_pred, alpha=0.7)
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Random Forest: Actual vs Predicted CO₂ Emissions")
plt.grid(True)
plt.tight_layout()
plt.show()

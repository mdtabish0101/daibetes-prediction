import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge  # Regularized model
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Ridge regression model (L2 Regularization)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open("../model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model training complete! Model saved in 'model' folder.")

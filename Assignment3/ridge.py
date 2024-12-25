import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd

num_features = int(input("Enter the number of features: "))
num_data_points = int(input("Enter the number of data points: "))
lambda_val = float(input("Enter the regularization parameter (lambda): "))


print("Enter the data for features (space-separated values for each data point):")
data = []
for i in range(num_data_points):
    row = list(map(float, input(f"Data point {i+1}: ").split()))
    data.append(row)


data = np.array(data)

# Separate features and target variable
X = data[:, :-1]  # Features (all columns except last one)
y = data[:, -1]   # Target variable (last column)


ridge_model = Ridge(alpha=lambda_val)
ridge_model.fit(X, y)

# Displaying results
print(f"Coefficients (weights): {ridge_model.coef_}")
print(f"Intercept: {ridge_model.intercept_}")

# Prediction using the learned model
predictions = ridge_model.predict(X)
print(f"Predictions: {predictions}")

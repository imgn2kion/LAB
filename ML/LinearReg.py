import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([1.5, 3.0, 4.5, 6.1, 7.8, 9.1, 10.5, 12.2, 14.1, 15.8])

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

slope = model.coef_[0]
intercept = model.intercept_

mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("Model Parameters:")
print(f"Slope (Coefficient): {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

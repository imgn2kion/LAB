import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
x2 = np.array([2.0, 2.5, 3.0, 3.5, 4.0]).reshape(-1, 1)
y = np.array([2.5, 3.0, 4.5, 5.0, 6.5])

model1 = LinearRegression()
model1.fit(x1, y)

b0 = model1.intercept_
b1 = model1.coef_

print(f"Linear Regression Coefficients: b0 = {b0}, b1 = {b1}")

prediction_x1 = b0 + b1 * 4.0
print(f"Linear Regression Prediction for x1 = 4.0: {prediction_x1}")

X = np.concatenate((x1, x2), axis=1)
model2 = LinearRegression()
model2.fit(X, y)

b0 = model2.intercept_
b1 = model2.coef_[0]
b2 = model2.coef_[1]

print(f"Multiple Linear Regression Coefficients: b0 = {b0}, b1 = {b1}, b2 = {b2}")

prediction_x1_x2 = b0 + b1 * 4.0 + b2 * 3.0
print(f"Multiple Linear Regression Prediction for x1 = 4.0, x2 = 3.0: {prediction_x1_x2}")

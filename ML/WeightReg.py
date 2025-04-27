import numpy as np
import pandas as pd

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
Y = np.array([1.2, 2.8, 3.6, 4.5, 5.1])

r = 1.0
x0 = np.array([1.5, 2.5, 3.5, 4.5])

def weight_matrix(x, x0, r):
    return np.exp(-((x - x0) ** 2) / (2 * r ** 2))

def beta(X, Y, x0, r):
    W = np.diag(weight_matrix(X.flatten(), x0, r))
    X_bias = np.c_[np.ones(X.shape[0]), X]
    theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ Y)
    return theta

predictions = []
for x_query in x0:
    θ = beta(X, Y, x_query, r)
    prediction = θ[0] + θ[1] * x_query
    predictions.append(prediction)

df = pd.DataFrame({'Query Point': x0, 'Predicted Value': predictions})
print(df)

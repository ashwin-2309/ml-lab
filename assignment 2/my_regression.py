import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add a column of ones to X for the bias term
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate the weights using the normal equation
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Add a column of ones to X for the bias term
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate the predicted y values using the weights
        y_pred = X @ self.weights
        return y_pred

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

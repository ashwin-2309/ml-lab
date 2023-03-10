import numpy as np


class LinearModel:
    """Base class for linear models"""

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class LinearRegression(LinearModel):
    """Linear regression model"""

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.weights

    def accuracy(self, y_pred, y_true):
        """Calculates the accuracy of the model's predictions"""
        return np.mean(y_pred == y_true)


class ErrorMetrics(LinearModel):
    """Class for calculating error metrics of linear models"""

    def __init__(self):
        super().__init__()

    def mse(self, y_pred, y_true):
        """Calculates the mean squared error"""
        return np.mean((y_pred - y_true)**2)

import numpy as np


class Model:
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def accuracy(self, y_pred, y_true):
        pass


class LinearRegression(Model):
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights

    def accuracy(self, y_pred, y_true):
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_mean)**2)
        ss_res = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_res / ss_tot)

        return r2


class DataGenerator:
    def sine_wave(self, n):

        X_train = np.random.random(n)
        y_train = np.sin(X_train) + np.random.random(n) * 0.1
        return X_train, y_train


if __name__ == "__main__":

    np.random.seed(0)
    data_gen = DataGenerator()
    X_train, y_train = data_gen.sine_wave(100)
    X_train = X_train.reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)

    print("Mean Squared Error: ", reg.mse(y_pred, y_train))
    print("Accuracy: ", reg.accuracy(y_pred, y_train))

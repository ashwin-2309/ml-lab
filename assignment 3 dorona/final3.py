import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class DiseaseSpread:
    def __init__(self):
        self.years = np.arange(1900, 2023)
        self.D_0 = 100
        self.D = self.D_0 * (2.4)**((self.years-1900)/3) + 0
        self.train_test_split(0.2)

    def train_test_split(self, test_size):
        idx = int(len(self.years)*(1-test_size))
        self.train_data = self.years[:idx].reshape(-1, 1)
        self.test_data = self.years[idx:].reshape(-1, 1)
        self.train_target = self.D[:idx]
        self.test_target = self.D[idx:]

    def linear_regression(self):
        reg = LinearRegression().fit(self.train_data, self.train_target)
        self.train_pred = reg.predict(self.train_data)
        self.test_pred = reg.predict(self.test_data)
        self.train_mse = mean_squared_error(self.train_target, self.train_pred)
        self.test_mse = mean_squared_error(self.test_target, self.test_pred)

        print('Linear Regression - Training MSE:', self.train_mse)
        print('Linear Regression - Testing MSE:', self.test_mse)

        plt.scatter(self.years, self.D, label='Actual data')
        plt.plot(self.years, self.train_pred, label='Linear Regression')
        plt.legend()
        plt.title('Linear Regression')
        plt.show()

    def log_regression(self):
        log_D = np.log(self.D)
        self.train_data, self.test_data, self.train_target, self.test_target = self.train_test_split(
            0.2)
        reg = LinearRegression().fit(self.train_data, self.train_target)
        self.train_pred = reg.predict(self.train_data)
        self.test_pred = reg.predict(self.test_data)
        self.train_mse = mean_squared_error(self.train_target, self.train_pred)
        self.test_mse = mean_squared_error(self.test_target, self.test_pred)

        print('Log + Linear Regression - Training MSE:', self.train_mse)
        print('Log + Linear Regression - Testing MSE:', self.test_mse)

        plt.scatter(self.years, log_D, label='Actual data')
        plt.plot(self.years, self.train_pred, label='Log+ Linear Regression')
        plt.legend()
        plt.title('Log + Linear Regression')
        plt.show()


if __name__ == "__main__":
    ds = DiseaseSpread()
    ds.linear_regression()
    ds.log_regression()

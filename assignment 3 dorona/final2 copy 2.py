import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Prepare data starting from 1900 to 2023
years = np.arange(1900, 2023)

D_0 = 100

D = D_0 * (2.4)**((years-1900)/3) + np.random.normal(loc=0, scale=1, size=123)
# log transformation
D = np.log(D)
# Normalization
scaler = MinMaxScaler()
D = scaler.fit_transform(D.reshape(-1, 1))

# Make two disjoint sets, "Training set" and "Testing set," with 80% of training data and 20% of testing data
train_data, test_data, train_target, test_target = train_test_split(
    years.reshape(-1, 1), D.reshape(-1, 1), test_size=0.2)

# Linear Regression
reg = LinearRegression().fit(train_data, train_target)
train_pred = reg.predict(train_data)
test_pred = reg.predict(test_data)
train_mse = mean_squared_error(train_target, train_pred)
test_mse = mean_squared_error(test_target, test_pred)

print('Linear Regression - Training MSE:', train_mse)
print('Linear Regression - Testing MSE:', test_mse)

# Plot the data with the model
plt.scatter(years, D, label='Actual data')
plt.plot(years, reg.predict(
    years.reshape(-1, 1)), label='Linear Regression')
plt.legend()
plt.title('Linear Regression')
plt.show()

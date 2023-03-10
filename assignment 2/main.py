# main.py
import numpy as np
from my_regression import LinearRegression

# generate some random data
np.random.seed(0)
X_train = np.random.rand(100, 4)
y_train = X_train @ np.random.rand(4)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)

print("Mean Squared Error: ", reg.mse(y_pred, y_train))

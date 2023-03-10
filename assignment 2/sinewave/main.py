from linear_regression import LinearModel
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)

print("Mean Squared Error: ", ErrorMetrics().mse(y_pred, y_train))
print("Accuracy: ", reg.accuracy(y_pred, y_train))

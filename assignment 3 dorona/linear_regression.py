import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data_preparation import prepare_data


def linear_regression(start_year, end_year, pct_train):
    train_time, test_time, train_log_d, test_log_d = prepare_data(
        start_year, end_year, pct_train)

    # fit the linear regression model
    reg = LinearRegression().fit(train_time.reshape(-1, 1), train_log_d)

    # predict the target values for the training and testing sets
    train_log_d_pred = reg.predict(train_time.reshape(-1, 1))
    test_log_d_pred = reg.predict(test_time.reshape(-1, 1))

    # calculate the training and testing accuracy
    train_accuracy = reg.score(train_time.reshape(-1, 1), train_log_d)
    test_accuracy = reg.score(test_time.reshape(-1, 1), test_log_d)

    # plot the data with the model
    plt.scatter(train_time, train_log_d, label="Training Data")
    plt.scatter(test_time, test_log_d, label="Testing Data")
    plt.plot(time, reg.predict(time.reshape(-1, 1)),
             color='red', label="Regression Line")
    plt.xlabel("Year")
    plt.ylabel("Log of Dorona cases")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

    # print the training and testing accuracy
    print("Training accuracy:", train_accuracy)
    print("Testing accuracy:", test_accuracy)

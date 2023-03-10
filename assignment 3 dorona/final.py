import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def log_regression(start_year, end_year, pct_train):
    # Define the years range
    years = np.arange(start_year, end_year + 1)
    # Create the time variable by subtracting 1900 from each year
    time = years - 1900
    # Generate the target variable with a exponential function and random noise
    d = 100 * (2.4) ** time + np.random.normal(0, 1, len(time))
    # Take the log of the target variable to create the log_d variable
    log_d = np.log(d)
    # Split the time and log_d variables into training and testing sets based on the pct_train value
    train_size = int(pct_train * len(time))
    train_time = time[:train_size]
    test_time = time[train_size:]
    train_log_d = log_d[:train_size]
    test_log_d = log_d[train_size:]

    # Fit the log regression model using the training data
    reg = LinearRegression().fit(train_time.reshape(-1, 1), train_log_d)

    # Predict the target values for the training and testing sets
    train_log_d_pred = reg.predict(train_time.reshape(-1, 1))
    test_log_d_pred = reg.predict(test_time.reshape(-1, 1))

    # Calculate the training and testing accuracy
    train_accuracy = reg.score(train_time.reshape(-1, 1), train_log_d)
    test_accuracy = reg.score(test_time.reshape(-1, 1), test_log_d)

    # Plot the data with the model
    plt.scatter(train_time, train_log_d, label="Training Data")
    plt.scatter(test_time, test_log_d, label="Testing Data")
    plt.plot(time, reg.predict(time.reshape(-1, 1)),
             color='red', label="Regression Line")
    plt.xlabel("Year")
    plt.ylabel("Log of Dorona cases")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

    # Print the training and testing accuracy
    print("Training accuracy:", train_accuracy)
    print("Testing accuracy:", test_accuracy)


# Call the function with the desired start and end years and training percentage
log_regression(2020, 2023, 0.8)

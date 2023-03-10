# class for logistic regression for the following algorithms:
# ❖ Find the “Log Likelihood loss”.
# ❖ Maximize the loss function with the help of :
# ➢ “Momentum based GDA”
# ➢ “Nesterov Accelerated GDA”
# ➢ “Line Search GDA”
# ➢ “Ada GDA”
# ➢ “RMPprop GDA”
# ➢ “Adam GDA”
# log loss function

# accuracy_score
# plot loss
# plot accuracy
# print weights
# print loss
# print accuracy

# import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

# class for logistic regression


class LogisticRegressionGD(object):
    def __init__(self):
        # ask for input of which algorithm to use for gradient descent and set that to self.algorithm
        self.algorithm = int(input(
            "Enter the algorithm to use for gradient descent: \n1. Momentum based GDA \n2. Nesterov Accelerated GDA \n3. Line Search GDA \n4. Ada GDA \n5. RMPprop GDA \n6. Adam GDA \n"))

    def softmax(self, z):
        # softmax function
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    # log loss function
    def log_loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat))

    # accuracy score function
    def accuracy_score(self, y, y_hat):
        return accuracy_score(y, y_hat)

    # plot loss function
    def plot_loss(self, loss):
        plt.plot(loss)
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    # function for momentum gradient descent
    def momentum_gradient_descent(self, X, y, learning_rate=0.01, epochs=1500, beta=1.5):
        # initialize weights
        self.weights = np.random.randn(X.shape[1], y.shape[1])
        # initialize bias
        self.bias = np.random.randn(1, y.shape[1])
        # initialize loss
        loss = []
        # initialize accuracy
        accuracy = []
        # initialize velocity
        velocity = 0
        # loop for number of epochs
        for i in range(epochs):
            # calculate y_hat
            y_hat = self.softmax(np.dot(X, self.weights) + self.bias)
            # calculate loss
            loss.append(self.log_loss(y, y_hat))
            # calculate accuracy
            accuracy.append(self.accuracy_score(y, y_hat))
            # calculate gradients
            dw = np.dot(X.T, (y_hat - y))
            db = np.sum(y_hat - y, axis=0, keepdims=True)
            # update weights and bias
            velocity = beta * velocity + learning_rate * dw
            self.weights = self.weights - velocity
            self.bias = self.bias - learning_rate * db
        # plot loss
        self.plot_loss(loss)
        # print loss
        print("Loss: ", loss[-1])
        # print accuracy
        print("Accuracy: ", accuracy[-1])

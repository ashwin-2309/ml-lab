# Class to implement a logistic regression model and its methods
# Import the required librearies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

# model class


class LogisticRegressionModel:
    # constructor
    def __init__(self):
        # Ask user to choose variant of gradient descent
        print("Choose the variant of gradient descent")
        print("1. Stochastic Gradient Descent")
        print("2. Batch Gradient Descent")
        print("3. Mini Batch Gradient Descent")
        print("4. Newton Raphson")
        self.choice = int(input("Enter your choice: "))
        # create the model according to the choice
        if self.choice == 1:
            print("Using Stochastic Gradient Descent")
            # create the model
            self.model = LogisticRegression(
                solver='sag', max_iter=1, warm_start=True, random_state=42)
        elif self.choice == 2:
            print("Using Batch Gradient Descent")
            # create the model
            self.model = LogisticRegression(
                solver='lbfgs', max_iter=1, warm_start=True, random_state=42)
        elif self.choice == 3:
            print("Using Mini Batch Gradient Descent")
            # create the model
            self.model = LogisticRegression(
                solver='saga', max_iter=1, warm_start=True, random_state=42)
        elif self.choice == 4:
            print("Using Newton Raphson")
            # create the model
            self.model = LogisticRegression(
                solver='newton-cg', max_iter=1, warm_start=True, random_state=42)
        else:
            print("Invalid choice")
            exit()

        # Train the model for epochs and store the loss and accuracy
    def train(self, X_train, X_test, y_train, y_test, epochs=150):
        loss = []
        accuracy = []
        for i in range(epochs):
            if self.choice == 3:
                # fit in batches of 10
                for j in range(0, X_train.shape[0], 10):
                    self.model.fit(X_train[j:j+10], y_train[j:j+10])
            else:
                self.model.fit(X_train, y_train)
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            accuracy.append([train_accuracy, test_accuracy])
            y_train_pred_proba = self.model.predict_proba(X_train)
            y_test_pred_proba = self.model.predict_proba(X_test)
            train_loss = log_loss(y_train, y_train_pred_proba)
            test_loss = log_loss(y_test, y_test_pred_proba)
            loss.append([train_loss, test_loss])
        self.loss = loss
        self.accuracy = accuracy
    # plot the loss vs epochs graph

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'])
        plt.show()
    # plot the accuracy vs epochs graph

    def plot_accuracy(self):
        plt.plot(self.accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'])
        plt.show()
    # print the optimized weights and bias

    def print_weights(self):
        print("Optimized weights: ", self.model.coef_)
        print("Optimized bias: ", self.model.intercept_)
    # predict the final loss and accuracy

    def predict(self, X_test, y_test):
        print("Final loss on training set: ", self.loss[-1][0])
        print("Final loss on test set: ", self.loss[-1][1])
        print("Final accuracy on training set: ", self.accuracy[-1][0])
        print("Final accuracy on test set: ", self.accuracy[-1][1])

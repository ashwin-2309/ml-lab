#create a model class and data class
# ❖ Prob-1 : Consider the famous “Iris” dataset of sklearn library and you are supposed
# to do following experiments on it:
# ❑ Design a “simple logistic regression with two class problem”
# ❑ Design a “multiple logistic regression with taking two class problem”
# ❑ Design a “multiple logistic regression with all classes problem”
# Apply the followings on each of the above model :
# ❖ Find the “Log Likelihood loss”.
# ❖ Maximize the loss function with the help of :
# ➢ “Stochastic Gradient Decent”
# ➢ “Batched Gradient Decent”
# ➢ “Mini Batched Gradient Decent”
# ➢ “Newton Raphson”
# Lab Assignment – 4 (Logistic regression)

# ❖ Plot the respective loss vs epochs (Iteration) graph as well as accuracy vs epochs
# graph for both training and testing samples for each optimization process. Also print
# the optimized weights and bias.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


class Model:
    # constructor to store the data
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    # function to create the model
    def create_model(self):
        # ask user to choose variant of gradient descent
        print("Choose the variant of gradient descent")
        print("1. Stochastic Gradient Descent")
        print("2. Batch Gradient Descent")
        print("3. Mini Batch Gradient Descent")
        print("4. Newton Raphson")
        self.choice = int(input("Enter your choice: "))
        # create the model according to the choice
        if self.choice == 1:
            model = LogisticRegression(solver='sag', max_iter=1000)
        elif self.choice == 2:
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif self.choice == 3:
            model = LogisticRegression(solver='saga', max_iter=1000)
        elif self.choice == 4:
            model = LogisticRegression(solver='newton-cg', max_iter=1000)

    # function to train the model
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    # function to predict the model
    def predict_model(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_train_pred_prob = self.model.predict_proba(self.X_train)
        self.y_test_pred_prob = self.model.predict_proba(self.X_test)

    # function to calculate the training and testing accuracy
    # and print it
    def accuracy(self):
        print("Training Accuracy: ", accuracy_score(self.y_train, self.y_train_pred))
        print("Testing Accuracy: ", accuracy_score(self.y_test, self.y_test_pred))

    # function to calculate the training and testing log loss
    # and print it
    def log_loss(self):
        print("Training Log Loss: ", log_loss(self.y_train, self.y_train_pred_prob))
        print("Testing Log Loss: ", log_loss(self.y_test, self.y_test_pred_prob))



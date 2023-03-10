import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from dataset_iris import IrisData, TwoClassData, MultiClassData


class Model:
    # constructor to store the data
    def __init__(self, data_class):
        self.data = data_class

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
            self.model = LogisticRegression(
                solver='sag', max_iter=1, warm_start=True)
        elif self.choice == 2:
            self.model = LogisticRegression(
                solver='lbfgs', max_iter=1, warm_start=True)
        elif self.choice == 3:
            self.model = LogisticRegression(
                solver='saga', max_iter=1, warm_start=True)
        elif self.choice == 4:
            self.model = LogisticRegression(
                solver='newton-cg', max_iter=1, warm_start=True)

    # function to train the model
    def train_model(self, num_iterations=100, batch_size=10):
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

        if self.choice == 2:
            num_batches = 10
            for i in range(num_iterations):
                for j in range(num_batches):
                    start_index = j*batch_size
                    end_index = min((j+1)*batch_size,
                                    self.data.X_train.shape[0])
                    X_batch = self.data.X_train[start_index:end_index, :]
                    y_batch = self.data.y_train[start_index:end_index]
                    self.model.fit(X_batch, y_batch)
                self.loss_train.append(
                    log_loss(self.data.y_train, self.model.predict_proba(self.data.X_train)))
                self.loss_test.append(
                    log_loss(self.data.y_test, self.model.predict_proba(self.data.X_test)))
                self.acc_train.append(accuracy_score(
                    self.data.y_train, self.model.predict(self.data.X_train)))
                self.acc_test.append(accuracy_score(
                    self.data.y_test, self.model.predict(self.data.X_test)))

        else:
            for i in range(num_iterations):
                self.model.fit(self.data.X_train, self.data.y_train)
                self.loss_train.append(
                    log_loss(self.data.y_train, self.model.predict_proba(self.data.X_train)))
                self.loss_test.append(
                    log_loss(self.data.y_test, self.model.predict_proba(self.data.X_test)))
                self.acc_train.append(accuracy_score(
                    self.data.y_train, self.model.predict(self.data.X_train)))
                self.acc_test.append(accuracy_score(
                    self.data.y_test, self.model.predict(self.data.X_test)))

    # function to predict the model
    def predict_model(self):
        self.y_train_pred = self.model.predict(self.data.X_train)
        self.y_test_pred = self.model.predict(self.data.X_test)
        self.y_train_pred_prob = self.model.predict_proba(self.data.X_train)
        self.y_test_pred_prob = self.model.predict_proba(self.data.X_test)

    # function to calculate the training and testing accuracy
    # and print it

    def accuracy(self):
        print("Training Accuracy: ", self.loss_train[-1])
        print("Testing Accuracy: ", self.loss_test[-1])

    # function to calculate the training and testing log loss
    # and print it

    def log_loss(self):
        print("Training Log Loss: ", self.loss_train[-1])
        print("Testing Log Loss: ", self.loss_test[-1])

    # function to plot the loss vs epochs and accuracy vs epochs graph

    def plot(self):
        # plot the loss vs epochs graph
        plt.plot(self.loss_train, label='Training Loss')
        plt.plot(self.loss_test, label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # plot the accuracy vs epochs graph
        plt.plot(self.acc_train, label='Training Accuracy')
        plt.plot(self.acc_test, label='Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # print the optimized weights and bias
        print("Optimized Weights: ", self.model.coef_)
        print("Optimized Bias: ", self.model.intercept_)

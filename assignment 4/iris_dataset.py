# Dataset class to import the iris dataset,preprocess it according to requirement,
# split into train and test sets and return the data in the numpy array format
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class IrisDataset:
    # Load the iris dataset
    def __init__(self):
        self.iris = datasets.load_iris()
    # For simple logistic regression with taking 2 class problem
    # Reduce the dataset to 2 feature and 2 classes

    def convert_to_simple_binary(self):
        x = self.iris.data[:, :1]
        y = self.iris.target
        # Delete the 3rd class
        x = np.delete(x, np.where(y == 2), axis=0)
        y = np.delete(y, np.where(y == 2), axis=0)
        # Store
        self.x = x
        self.y = y
    # For multiple logistic regression with taking 2 class problem
    # Reduce the dataset to 2 classes

    def convert_to_multiple_binary(self):
        x = self.iris.data
        y = self.iris.target
        # Delete the 3rd class
        x = np.delete(x, np.where(y == 2), axis=0)
        y = np.delete(y, np.where(y == 2), axis=0)
        # Store
        self.x = x
        self.y = y
    # For multiple logistic regression with taking all classes problem

    def convert_to_multiple(self):
        x = self.iris.data
        y = self.iris.target
        # Store
        self.x = x
        self.y = y

    def split_dataset(self, test_size=0.2, random_state=42):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)

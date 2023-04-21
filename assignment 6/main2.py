import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the mnist dataset from OpenML
mnist = fetch_openml('mnist_784')

# get the images and labels
X, y = mnist['data'], mnist['target']

# convert data type to float32
X = X.astype('float32')

# normalize the pixel values between 0 and 1
X /= 255.0

# split data into train and test sets
split = 60000  # 60k training images, 10k test images
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


def pca_scratch(n_components):
    # calculate the covariance matrix
    cov_matrix = np.cov(X_train.T)
    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # select the first n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # calculate the principal components
    principal_components = np.dot(X_train, eigenvectors)

    # calculate the reconstructed data
    reconstructed_data = np.dot(principal_components, eigenvectors.T)

    # calculate the mse
    mse = ((X_train - reconstructed_data) ** 2).mean(axis=None)

    # Train the logistic regression model with the principal components
    logistic_regression = LogisticRegression()
    logistic_regression.fit(principal_components, y_train)

    # Predict the labels of the test data
    y_pred_pca = logistic_regression.predict(np.dot(X_test, eigenvectors))

    # Calculate the accuracy score
    accuracy_pca = accuracy_score(y_test, y_pred_pca)

    # print mse, accuracy and variance
    print(
        f"MSE with {n_components} principal components using eigendecomposition method: {mse}")
    print(
        f"Accuracy with {n_components} principal components using eigendecomposition method : {accuracy_pca}")
    print(
        f"Variance with {n_components} principal components using eigendecomposition method: {eigenvalues.sum() / np.trace(cov_matrix)}")

    # Show the first principal component in image format
    plt.imshow(eigenvectors[:, 0].reshape(28, 28))
    plt.show()


def train_logistic_regression_with_pca(n_components):
    # create the pca object with n_components
    pca = PCA(n_components=n_components)

    # fit the pca object to the data
    pca.fit(X_train)

    # transform the data
    X_train_pca = pca.transform(X_train)

    # create the logistic regression object
    logistic_regression = LogisticRegression()

    # fit the logistic regression object to the data
    logistic_regression.fit(X_train_pca, y_train)

    # predict the labels of the test data
    y_pred_pca = logistic_regression.predict(pca.transform(X_test))

    # calculate the accuracy score
    accuracy_pca = accuracy_score(y_test, y_pred_pca)

    # print the accuracy score
    print(f"Accuracy with {n_components} principal components: {accuracy_pca}")

    # show the principal components in image format
    # for i in range(n_components):
    plt.imshow(pca.components_.reshape(n_components, 28, 28)[0])
    plt.show()

    # Compute the original feature space for each of the values of M.
    X_train_reconstructed = pca.inverse_transform(X_train_pca)

    # Compute the Mean Square Error (MSE) for each of the cases.
    mse = ((X_train - X_train_reconstructed) ** 2).mean(axis=None)
    print(f"MSE with {n_components} principal components: {mse}")
    # print the mse and accuracy
    print(f"MSE with {n_components} principal components: {mse}")
    print(f"Accuracy with {n_components} principal components: {accuracy_pca}")
    # variance with n_components
    print(
        f"Variance with {n_components} principal components: {pca.explained_variance_ratio_.sum()}")

# build a pca function with n_components as parameter and its name should be pca_scratch


# call the function with 10 principal components
# train_logistic_regression_with_pca(100)
pca_scratch(n_components=100)

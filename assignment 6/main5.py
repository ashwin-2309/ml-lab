import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense

# Load the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784')

# Get the images and labels
X, y = mnist['data'], mnist['target']

# Convert data type to float32
X = X.astype('float32')

# Normalize the pixel values between 0 and 1
X /= 255.0

# Split data into train and test sets
split = 60000  # 60k training images, 10k test images
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create a function to display the principal components


def show_principal_components(pca, n_components):
    fig, axes = plt.subplots(2, (n_components+1)//2, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        if i < n_components:
            ax.imshow(pca.components_[i].reshape((28, 28)), cmap='gray')
            ax.set(title=f'PC {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()


# Create a function to perform PCA on MNIST dataset


def perform_pca(n_components):
    # Create the PCA object with n_components
    pca = PCA(n_components=n_components)

    # Fit the PCA object to the data
    pca.fit(X_train)

    # Display the principal components
    show_principal_components(pca, n_components)

    # Transform the data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Compute the original feature space for each of the values of M
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    X_test_reconstructed = pca.inverse_transform(X_test_pca)

    # Compute the Mean Square Error (MSE) for each of the cases
    mse_train = mean_squared_error(X_train, X_train_reconstructed)
    mse_test = mean_squared_error(X_test, X_test_reconstructed)

    print("Number of principal components:", n_components)
    print(f"MSE on train set: {mse_train:.2f}")
    print(f"MSE on test set: {mse_test:.2f}")


# Perform PCA with different number of components
n_components_list = [5, 10, 25, 64, 100, 200, 784]
for n_components in n_components_list:
    perform_pca(n_components)

# Define the parameters of the autoencoder
P, Q, R, M_autoencoder = 256, 128, 64, 30

# Create the autoencoder model
input_img = Input(shape=(784,))
encoded = Dense(P, activation='relu')(input_img)
encoded = Dense(Q, activation='relu')(encoded)
encoded = Dense(R, activation='relu')(encoded)
encoded = Dense(M_autoencoder, activation='relu')(encoded)
decoded = Dense(R, activation='relu')(encoded)
decoded = Dense(Q, activation='relu')(decoded)
decoded = Dense(P, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256,
                shuffle=True, validation_data=(X_test, X_test))

# Compute the original feature space for each of the values of M
X_train_reconstructed_autoencoder = autoencoder.predict(X_train)
X_test_reconstructed_autoencoder = autoencoder.predict(X_test)

# Compute the Mean Square Error (MSE) for each of the cases
mse_train_autoencoder = mean_squared_error(
    X_train, X_train_reconstructed_autoencoder)
mse_test_autoencoder = mean_squared_error(
    X_test, X_test_reconstructed_autoencoder)

# Compare the performance of PCA and Autoencoder in terms of MSE
print("Comparison of PCA and Autoencoder performance:")
print(f"MSE on train set with PCA: {mse_train:.2f}")
print(f"MSE on test set with PCA: {mse_test:.2f}")
print(f"MSE on train set with Autoencoder: {mse_train_autoencoder:.2f}")
print(f"MSE on test set with Autoencoder: {mse_test_autoencoder:.2f}")

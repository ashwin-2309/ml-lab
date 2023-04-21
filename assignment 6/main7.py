from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser="auto")
X = mnist.data
y = mnist.target.astype(np.uint8)

# Normalize the data
X = X / 255.0

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Task 1: Apply PCA on the training set for different values of M


def apply_pca(X_train, M):
    # Apply PCA with M components
    pca = PCA(n_components=M)
    pca.fit(X_train)

    # Transform the training set to M dimensions
    X_train_transformed = pca.transform(X_train)

    return X_train_transformed


Ms = [5, 10, 25, 64, 100, 200, 784]
X_train_transformed = []

for M in Ms:
    X_train_transformed.append(apply_pca(X_train, M))

# Task 2: Show principal components for each value of M


def show_principal_components(pca, M):
    fig, axes = plt.subplots(1, M, figsize=(10, 10))

    for i in range(M):
        # Reshape the eigenvector to a 28x28 image
        component = pca.components_[i].reshape(28, 28)

        axes[i].imshow(component, cmap='gray')
        axes[i].axis('off')

    plt.show()


for i, M in enumerate(Ms):
    pca = PCA(n_components=M)
    pca.fit(X_train)
    show_principal_components(pca, M)

# Task 3: Compute the original feature space for each value of M and compute the Mean Square Error (MSE)


def reconstruct_images(X_transformed, pca):
    X_reconstructed = pca.inverse_transform(X_transformed)
    return X_reconstructed


def compute_mse(X, X_reconstructed):
    mse = np.mean(np.power(X - X_reconstructed, 2))
    return mse


MSEs = []

for i, M in enumerate(Ms):
    pca = PCA(n_components=M)
    pca.fit(X_train)

    X_train_transformed = apply_pca(X_train, M)
    X_train_reconstructed = reconstruct_images(X_train_transformed, pca)
    mse_train = compute_mse(X_train, X_train_reconstructed)

    X_test_transformed = pca.transform(X_test)
    X_test_reconstructed = reconstruct_images(X_test_transformed, pca)
    mse_test = compute_mse(X_test, X_test_reconstructed)

    MSEs.append((M, mse_train, mse_test))

    print(f"M = {M}")
    print(f"MSE (train) = {mse_train:.4f}")
    print(f"MSE (test) = {mse_test:.4f}")

# Task 4: Construct an autoencoder with dimensions (784-P-Q-R-M)


def create_autoencoder(P, Q, R, M):
    # Encoder
    input_img = Input(shape=(784,))
    encoded = Dense(P, activation='relu')(input_img)
    encoded = Dense(Q, activation='relu')(encoded)
    encoded = Dense(R, activation='relu')(encoded)
    encoded = Dense(M, activation='relu')(encoded)

    # Decoder
    decoded = Dense(R, activation='relu')(encoded)
    decoded = Dense(Q, activation='relu')(decoded)
    decoded = Dense(P, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    # Autoencoder
    autoencoder = Model(input_img, decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                    shuffle=True, validation_data=(X_test, X_test))

    return autoencoder


Ps = [50, 100, 200, 300, 500]
Qs = [50, 100, 200, 300, 500]
Rs = [50, 100, 200, 300, 500]
Ms = [5, 10, 25, 64, 100, 200, 784]

# Train the autoencoder for each combination of P, Q, R, and M
autoencoders = {}
for P in Ps:
    for Q in Qs:
        for R in Rs:
            for M in Ms:
                key = f"{P}-{Q}-{R}-{M}"
                autoencoder = create_autoencoder(P, Q, R, M)
                autoencoders[key] = autoencoder

# Compute the MSE for each autoencoder
autoencoder_MSEs = {}
for P in Ps:
    for Q in Qs:
        for R in Rs:
            for M in Ms:
                key = f"{P}-{Q}-{R}-{M}"
                autoencoder = autoencoders[key]

                # Compute the MSE for the training set
                X_train_reconstructed = autoencoder.predict(X_train)
                mse_train = compute_mse(X_train, X_train_reconstructed)

                # Compute the MSE for the test set
                X_test_reconstructed = autoencoder.predict(X_test)
                mse_test = compute_mse(X_test, X_test_reconstructed)

                autoencoder_MSEs[key] = (mse_train, mse_test)

                print(
                    f"{key}: MSE (train) = {mse_train:.4f}, MSE (test) = {mse_test:.4f}")

# Compare the performance of the autoencoder with PCA by plotting the MSEs for each value of M
pca_mse_train = [mse_train for M, mse_train, _ in MSEs]
pca_mse_test = [mse_test for M, _, mse_test in MSEs]

autoencoder_mse_train = []
autoencoder_mse_test = []

for P in Ps:
    for Q in Qs:
        for R in Rs:
            for M in Ms:
                key = f"{P}-{Q}-{R}-{M}"
                mse_train, mse_test = autoencoder_MSEs[key]
                autoencoder_mse_train.append(mse_train)
                autoencoder_mse_test.append(mse_test)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(Ms, pca_mse_train, label='PCA (train)')
axes[0].plot(Ms, pca_mse_test, label='PCA (test)')
axes[0].set_xlabel('M')
axes[0].set_ylabel('MSE')
axes[0].legend()

axes[1].scatter(autoencoder_mse_train, autoencoder_mse_test, alpha=0.5)
axes[1].set_xlabel('MSE (train)')
axes[1].set_ylabel('MSE (test)')

plt.show()

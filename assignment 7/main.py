import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True,
                     as_frame=False, data_home="./data", return_X_y=False, parser="auto")


# Convert the data array to float32
X = mnist.data.astype('float32')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X / 255, mnist.target.astype('float32'), test_size=0.2)

# Task 1: PCA
pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_pca, y_train)
accuracy3 = knn3.score(X_test_pca, y_test)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_pca, y_train)
accuracy5 = knn5.score(X_test_pca, y_test)
print('Task 1: PCA Accuracy (k=3):', accuracy3)
print('Task 1: PCA Accuracy (k=5):', accuracy5)

# Task 2: LDA
lda = LinearDiscriminantAnalysis(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_lda, y_train)
accuracy3 = knn3.score(X_test_lda, y_test)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_lda, y_train)
accuracy5 = knn5.score(X_test_lda, y_test)
print('Task 2: LDA Accuracy (k=3):', accuracy3)
print('Task 2: LDA Accuracy (k=5):', accuracy5)

# Task 3: Autoencoder
batch_size = 128
learning_rate = 0.001
n_epochs = 10
train_data = TensorDataset(torch.from_numpy(
    X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
autoencoder = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 9),
    nn.ReLU(),
    nn.Linear(9, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid())
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
for epoch in range(n_epochs):
    train_loss = 0.0
    for data, _ in train_loader:
        optimizer.zero_grad()
        recon = autoencoder(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Task 3: Autoencoder Epoch %d, Loss: %.6f' %
          (epoch + 1, train_loss / len(train_loader)))
with torch.no_grad():
    X_train_ae = autoencoder(torch.Tensor(X_train)).detach().numpy()
    X_test_ae = autoencoder(torch.Tensor(X_test)).detach().numpy()
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_ae[:, :9], y_train)
accuracy3 = knn3.score(X_test_ae[:, :9], y_test)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_ae[:, :9], y_train)
accuracy5 = knn5.score(X_test_ae[:, :9], y_test)
print('Task 3: Autoencoder Accuracy (k=3):', accuracy3)
print('Task 3: Autoencoder Accuracy (k=5):', accuracy5)

# Compare accuracies
print('Task 1: PCA Test Accuracy (k=3):', accuracy_score(
    y_test, knn3.predict(X_test_pca)))
print('Task 1: PCA Test Accuracy (k=5):', accuracy_score(
    y_test, knn5.predict(X_test_pca)))
print('Task 2: LDA Test Accuracy (k=3):',
      accuracy_score(y_test, knn3.predict(X_test_lda)))
print('Task 2: LDA Test Accuracy (k=5):',
      accuracy_score(y_test, knn5.predict(X_test_lda)))
print('Task 3: Autoencoder Test Accuracy (k=3):', accuracy_score(
    y_test, knn3.predict(X_test_ae[:, :9])))
print('Task 3: Autoencoder Test Accuracy (k=5):', accuracy_score(
    y_test, knn5.predict(X_test_ae[:, :9])))

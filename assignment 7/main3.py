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


class MNISTClassifier:
    def __init__(self):
        # Load the MNIST dataset
        self.mnist = fetch_openml('mnist_784', version=1, cache=True,
                                  as_frame=False, data_home="./data", return_X_y=True, parser="auto")

    def preprocess(self):
        # Convert the data array to float32
        X = self.mnist.data.astype('float32')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X / 255, self.mnist.target.astype('float32'), test_size=0.2)

        return X_train, X_test, y_train, y_test

    def run_pca(self, X_train, X_test, y_train, y_test):
        # Task 1: PCA
        self.pca = PCA(n_components=9)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        self.knn3 = KNeighborsClassifier(n_neighbors=3)
        self.knn3.fit(X_train_pca, y_train)
        self.accuracy3_pca = self.knn3.score(X_test_pca, y_test)
        self.knn5 = KNeighborsClassifier(n_neighbors=5)
        self.knn5.fit(X_train_pca, y_train)
        self.accuracy5_pca = self.knn5.score(X_test_pca, y_test)
        print('Task 1: PCA Accuracy (k=3):', self.accuracy3_pca)
        print('Task 1: PCA Accuracy (k=5):', self.accuracy5_pca)

    def run_lda(self, X_train, X_test, y_train, y_test):
        # Task 2: LDA
        self.lda = LinearDiscriminantAnalysis(n_components=9)
        X_train_lda = self.lda.fit_transform(X_train, y_train)
        X_test_lda = self.lda.transform(X_test)
        self.knn3 = KNeighborsClassifier(n_neighbors=3)
        self.knn3.fit(X_train_lda, y_train)
        self.accuracy3_lda = self.knn3.score(X_test_lda, y_test)
        self.knn5 = KNeighborsClassifier(n_neighbors=5)
        self.knn5.fit(X_train_lda, y_train)
        self.accuracy5_lda = self.knn5.score(X_test_lda, y_test)
        print('Task 2: LDA Accuracy (k=3):', self.accuracy3_lda)
        print('Task 2: LDA Accuracy (k=5):', self.accuracy5_lda)

    def run_autoencoder(self, X_train, X_test, y_train, y_test):
        # Task 3: Autoencoder
        batch_size = 128
        learning_rate = 0.001
        n_epochs = 10
        train_data = TensorDataset(torch.from_numpy(
            X_train), torch.from_numpy(X_train))      # use X_train as target instead of y_train
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.autoencoder = nn.Sequential(
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
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        for epoch in range(n_epochs):
            train_loss = 0.0
            for data, _ in train_loader:
                optimizer.zero_grad()
                recon = self.autoencoder(data)
                loss = criterion(recon, data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print('Task 3: Autoencoder Epoch %d, Loss: %.6f' %
                  (epoch + 1, train_loss / len(train_loader)))
        with torch.no_grad():
            X_train_ae = self.autoencoder(
                torch.Tensor(X_train)).detach().numpy()
            X_test_ae = self.autoencoder(torch.Tensor(X_test)).detach().numpy()
        self.knn3 = KNeighborsClassifier(n_neighbors=3)
        self.knn3.fit(X_train_ae[:, :9], y_train)
        self.accuracy3_ae = self.knn3.score(X_test_ae[:, :9], y_test)
        self.knn5 = KNeighborsClassifier(n_neighbors=5)
        self.knn5.fit(X_train_ae[:, :9], y_train)
        self.accuracy5_ae = self.knn5.score(X_test_ae[:, :9], y_test)
        print('Task 3: Autoencoder Accuracy (k=3):', self.accuracy3_ae)
        print('Task 3: Autoencoder Accuracy (k=5):', self.accuracy5_ae)

    def compare_accuracies(self, X_test, y_test):
        # Compare accuracies
        X_test_norm = (X_test / 255).astype('float32')     # normalize X_test
        print('Task 1: PCA Test Accuracy (k=3):', accuracy_score(
            y_test, self.knn3.predict(self.pca.transform(X_test_norm))))
        print('Task 1: PCA Test Accuracy (k=5):', accuracy_score(
            y_test, self.knn5.predict(self.pca.transform(X_test_norm))))
        print('Task 2: LDA Test Accuracy (k=3):', accuracy_score(
            y_test, self.knn3.predict(self.lda.transform(X_test_norm))))
        print('Task 2: LDA Test Accuracy (k=5):', accuracy_score(
            y_test, self.knn5.predict(self.lda.transform(X_test_norm))))
        X_test_ae_norm = (self.autoencoder(torch.Tensor(X_test_norm)).detach(
        ).numpy() / 255).astype('float32')    # normalize output of autoencoder
        print('Task 3: Autoencoder Test Accuracy (k=3):', accuracy_score(
            y_test, self.knn3.predict(X_test_ae_norm[:, :9])))
        print('Task 3: Autoencoder Test Accuracy (k=5):', accuracy_score(
            y_test, self.knn5.predict(X_test_ae_norm[:, :9])))

    def run_all(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.run_pca(X_train, X_test, y_train, y_test)
        self.run_lda(X_train, X_test, y_train, y_test)
        self.run_autoencoder(X_train, X_test, y_train, y_test)
        self.compare_accuracies(X_test, y_test)


mnist_classifier = MNISTClassifier()
mnist_classifier.run_all()

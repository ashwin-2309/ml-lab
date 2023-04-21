# import the mnist dataset from dataset.py
import matplotlib.pyplot as plt
from dataset import x_train, y_train, x_test, y_test

# i need to do pca on the data and then use the pca data to train the model

# create the model from scratch
from sklearn.decomposition import PCA
# import logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
# import the accuracy score from sklearn
from sklearn.metrics import accuracy_score

# create the pca object
# create 7 pca objects with 5,10,25,64,100,200,784 components
pca_5 = PCA(n_components=5)
pca_10 = PCA(n_components=10)
pca_25 = PCA(n_components=25)
pca_64 = PCA(n_components=64)
pca_100 = PCA(n_components=100)
pca_200 = PCA(n_components=200)
pca_784 = PCA(n_components=784)

# fit the pca object to the data
pca_5.fit(x_train)
pca_10.fit(x_train)
pca_25.fit(x_train)
pca_64.fit(x_train)
pca_100.fit(x_train)
pca_200.fit(x_train)
pca_784.fit(x_train)

# transform the data
x_train_pca_5 = pca_5.transform(x_train)
x_train_pca_10 = pca_10.transform(x_train)
x_train_pca_25 = pca_25.transform(x_train)
x_train_pca_64 = pca_64.transform(x_train)
x_train_pca_100 = pca_100.transform(x_train)
x_train_pca_200 = pca_200.transform(x_train)
x_train_pca_784 = pca_784.transform(x_train)

# create the logistic regression object
logistic_regression = LogisticRegression()

# fit the logistic regression object to the data
logistic_regression.fit(x_train_pca_5, y_train)
logistic_regression.fit(x_train_pca_10, y_train)
logistic_regression.fit(x_train_pca_25, y_train)
logistic_regression.fit(x_train_pca_64, y_train)
logistic_regression.fit(x_train_pca_100, y_train)
logistic_regression.fit(x_train_pca_200, y_train)
logistic_regression.fit(x_train_pca_784, y_train)

# predict the labels of the test data
y_pred_pca_5 = logistic_regression.predict(x_train_pca_5)
y_pred_pca_10 = logistic_regression.predict(x_train_pca_10)
y_pred_pca_25 = logistic_regression.predict(x_train_pca_25)
y_pred_pca_64 = logistic_regression.predict(x_train_pca_64)
y_pred_pca_100 = logistic_regression.predict(x_train_pca_100)
y_pred_pca_200 = logistic_regression.predict(x_train_pca_200)
y_pred_pca_784 = logistic_regression.predict(x_train_pca_784)

# calculate the accuracy score
accuracy_pca_5 = accuracy_score(y_train, y_pred_pca_5)
accuracy_pca_10 = accuracy_score(y_train, y_pred_pca_10)
accuracy_pca_25 = accuracy_score(y_train, y_pred_pca_25)
accuracy_pca_64 = accuracy_score(y_train, y_pred_pca_64)
accuracy_pca_100 = accuracy_score(y_train, y_pred_pca_100)
accuracy_pca_200 = accuracy_score(y_train, y_pred_pca_200)
accuracy_pca_784 = accuracy_score(y_train, y_pred_pca_784)

# print the accuracy score
print(accuracy_pca_5)
print(accuracy_pca_10)
print(accuracy_pca_25)
print(accuracy_pca_64)
print(accuracy_pca_100)
print(accuracy_pca_200)
print(accuracy_pca_784)

# for each pca component show the principal components in image format
# for 5 components
plt.imshow(pca_5.components_.reshape(5, 28, 28)[0])
plt.imshow(pca_5.components_.reshape(5, 28, 28)[1])
plt.imshow(pca_5.components_.reshape(5, 28, 28)[2])
plt.imshow(pca_5.components_.reshape(5, 28, 28)[3])
plt.imshow(pca_5.components_.reshape(5, 28, 28)[4])

# Compute the original feature space for each of the values of M. Also find the Mean
# Square Error (MSE) for each of the cases.

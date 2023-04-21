import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

# load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten the 28x28 images to a single vector of size 784
x_train_flatten = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flatten = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# set the dimensions for the encoded representation
encoding_dim = 32

# create the autoencoder model
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# compile the model with binary cross-entropy loss and the Adam optimizer
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train the model on the training set
autoencoder.fit(x_train_flatten, x_train_flatten,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_flatten, x_test_flatten))

# encode and decode some digits from the test set
encoded_imgs = autoencoder.predict(x_test_flatten)

# evaluate autoencoder performance
mse = np.square(np.subtract(encoded_imgs, x_test_flatten)).mean()
print("Mean squared error: {}".format(mse))

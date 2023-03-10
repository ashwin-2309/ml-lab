from iris_dataset import IrisDataset
from main import LogisticRegressionModel

# Load the dataset
iris = IrisDataset()
# Convert the dataset to simple binary
iris.convert_to_multiple_binary()
# Split the dataset into train and test
X_train, X_test, y_train, y_test = iris.split_dataset()
# Create the model
model = LogisticRegressionModel()
# Train the model for epochs and store the loss and accuracy
model.train(X_train, X_test, y_train, y_test, epochs=150)
# Plot the loss vs epochs graph
model.plot_loss()
# Plot the accuracy vs epochs graph
model.plot_accuracy()
# Print the optimized weights and bias
model.print_weights()
# Print the final loss and accuracy
model.predict(X_test, y_test)

from iris_dataset import IrisDataset
from main import LogisticRegressionModel

# Load the dataset
iris = IrisDataset()
# ask for user input
print("Enter 1 for simple binary logistic regression")
print("Enter 2 for multiple binary logistic regression")
print("Enter 3 for multiple logistic regression")
choice = int(input("Enter your choice: "))
# Convert the dataset to simple binary
if choice == 1:
    iris.convert_to_simple_binary()
elif choice == 2:
    iris.convert_to_multiple_binary()
elif choice == 3:
    iris.convert_to_multiple()
else:
    print("Invalid choice")
    exit()
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

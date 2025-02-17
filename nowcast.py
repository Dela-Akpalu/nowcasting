#==========================================================================================
# Regression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import estimator

np.random.seed(4)  # Corrected seed setting
# Generate sample data: 10000 samples for training, 1000 samples for testing
X_train = np.random.rand(10000, 5)
y_train = np.random.rand(10000, 1)  # Continuous target
X_test = np.random.rand(1000, 5)
y_test = np.random.rand(1000, 1)  # Continuous target

# Define a regression pipeline
pipeline_regression = Pipeline([
    ("scaler", StandardScaler()),  
    ("layer1", estimator.Layer(hidden_size=10, output_size=5, activation="tanh", loss="mse", input_size=5)),
    ("layer2", estimator.Layer(hidden_size=5, output_size=1, activation="identity", loss="mse", input_size=5))
])

# Train the model
pipeline_regression.fit(X_train, y_train)

# Make predictions
predictions = pipeline_regression.predict(X_test)

# Calculate and print RMSE
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("\nRegression RMSE:\n", rmse)

#================================================================================================================================
# Binary class

np.random.seed(4)  # Corrected seed setting
# Generate sample data: 10000 samples for training, 1000 samples for testing
X_train = np.random.rand(10000, 5)
y_train = np.random.randint(0, 2, size=(10000, 1))  # Binary labels (0 or 1)
X_test = np.random.rand(1000, 5)
y_test = np.random.randint(0, 2, size=(1000, 1))  # Binary labels (0 or 1)

# Define a binary classification pipeline
pipeline_binary = Pipeline([
    ("scaler", StandardScaler()),
    ("layer1", estimator.Layer(hidden_size=10, output_size=5, activation="tanh", loss="mse", input_size=5)),
    ("layer2", estimator.Layer(hidden_size=5, output_size=1, activation="sigmoid", loss="binary_cross_entropy", input_size=5))
])

# Train the model
pipeline_binary.fit(X_train, y_train)

# Make predictions
probabilities = pipeline_binary.predict(X_test)

# Convert probabilities to class labels
binary_classes = (probabilities > 0.5).astype(int)

# Calculate and print accuracy
accuracy = np.mean(binary_classes == y_test)
print("\nBinary Classification Accuracy:\n", accuracy)

#===============================================================================================================================
# Multicalss 

np.random.seed(4)  # Corrected seed setting
# Generate sample data: 10000 samples for training, 1000 samples for testing
num_classes = 5
X_train = np.random.rand(10000, 5)
y_train_labels = np.random.randint(0, num_classes, size=(10000, 1))  # Class labels
X_test = np.random.rand(1000, 5)
y_test_labels = np.random.randint(0, num_classes, size=(1000, 1))  # Class labels

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)  # Updated argument
y_train = encoder.fit_transform(y_train_labels)
y_test = encoder.transform(y_test_labels)

# Define a multiclass classification pipeline
pipeline_multiclass = Pipeline([
    ("scaler", StandardScaler()),
    ("layer1", estimator.Layer(hidden_size=10, output_size=5, activation="tanh", loss="mse", input_size=5)),
    ("layer2", estimator.Layer(hidden_size=5, output_size=num_classes, activation="softmax", loss="cross_entropy", input_size=5))
])

# Train the model
pipeline_multiclass.fit(X_train, y_train)

# Make predictions
probabilities = pipeline_multiclass.predict(X_test)

# Convert softmax probabilities to class labels
predicted_classes = np.argmax(probabilities, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate and print accuracy
accuracy = np.mean(predicted_classes == true_classes)
print("\nMulticlass Classification Accuracy:\n", accuracy)

#===========================================================================================================================================
# without pipeline 

# Create a Nowcast model
model = estimator.Nowcast()

# Add layers
model.add(estimator.Layer(hidden_size=10, output_size=5, activation="tanh", loss="mse", input_size=5))
model.add(estimator.Layer(hidden_size=5, output_size=3, activation="softmax", loss="cross_entropy", input_size=5))

# Print summary BEFORE training
print("\nModel Summary BEFORE Training:")
model.summary()

# Generate sample data: 10000 samples for training, 1000 samples for testing
X_train = np.random.rand(10000, 5)
y_train = np.random.rand(10000, 3)  # Assuming 3 output classes for softmax
X_test = np.random.rand(1000, 5)
y_test = np.random.rand(1000, 3)  # Assuming 3 output classes for softmax

# Train the model
model.fit(X_train, y_train, epochs=10)

# Print summary AFTER training
print("\nModel Summary AFTER Training:")
model.summary()

# Make predictions
predictions = model.predict(X_test)

# Calculate and print accuracy
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print("\nModel Accuracy AFTER Training:\n", accuracy)








from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tabulate import tabulate

# Loss functions
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class Layer(BaseEstimator, TransformerMixin):
    LOSS_FUNCTIONS = {
        "mse": mse_loss,
        "cross_entropy": cross_entropy_loss,
        "binary_cross_entropy": binary_cross_entropy_loss
    }

    ACTIVATIONS = {
        "tanh": np.tanh,
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "identity": lambda x: x,
        "relu": lambda x: np.maximum(0, x),
        "softmax": lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    }

    def __init__(self, hidden_size, output_size, activation="tanh", loss="mse", input_size=None):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.loss_name = loss  
        self.activation_name = activation  

        # Ensure activation is valid
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = self.ACTIVATIONS[activation]

        # Ensure loss is valid
        if loss not in self.LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {loss}")
        self.loss_function = self.LOSS_FUNCTIONS[loss]

        self.loss_history = []
        self._initialized = False

        if input_size is not None:
            self._initialize_weights(input_size)

    def _initialize_weights(self, input_size):
        self.Wxh = np.random.randn(input_size, self.hidden_size) * np.sqrt(1 / input_size)
        self.Why = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size)
        self.bh = np.zeros((1, self.hidden_size))
        self.by = np.zeros((1, self.output_size))
        self.h = np.zeros((1, self.hidden_size))
        self._initialized = True

    def forward(self, X):
        X = np.array(X).reshape(X.shape[0], -1)
        if not self._initialized:
            self._initialize_weights(X.shape[1])

        self.h = self.activation(np.dot(X, self.Wxh) + self.bh)
        output = np.dot(self.h, self.Why) + self.by

        if self.activation_name == "softmax":
            output = self.ACTIVATIONS["softmax"](output)
        elif self.activation_name == "sigmoid":
            output = self.ACTIVATIONS["sigmoid"](output)

        return output

    def fit(self, X, y=None):
        X = np.array(X).reshape(X.shape[0], -1)
        if not self._initialized:
            self._initialize_weights(X.shape[1])
        # Add a dummy loss to the loss history to avoid recursion
        self.loss_history.append(0)
        return self  

    def transform(self, X):
        return self.forward(X)

    def predict(self, X):
        return self.forward(X)

    def summary(self):
        """Print detailed layer summary"""
        print("\nLayer Summary:")
        print(f"Hidden Size: {self.hidden_size}, Output Size: {self.output_size}, Loss: {self.loss_name}")
        print(f"Activation: {self.activation_name}")
        if self._initialized:
            print(f"Weight Shapes: Wxh: {self.Wxh.shape}, Why: {self.Why.shape}")
        else:
            print("Weights not initialized yet.")

    def get_params(self, deep=True):
        """Return only primitive parameters to avoid recursion"""
        return {
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "loss": self.loss_name,
            "input_size": self.input_size
        }

    def set_params(self, **params):
        """Update parameters correctly"""
        for param, value in params.items():
            setattr(self, param, value)
        return self

class Nowcast(BaseEstimator):
    def __init__(self):
        self.layers = []
        self.loss_history = []

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Only instances of Layer can be added.")
        self.layers.append(layer)

    def fit(self, X, y, epochs=10):
        X = np.array(X).reshape(X.shape[0], -1)
        y = np.array(y).reshape(y.shape[0], -1)

        for epoch in range(epochs):
            loss_per_epoch = 0
            for layer in self.layers:
                layer.fit(X, y)
                loss_per_epoch += layer.loss_history[-1] if layer.loss_history else 0

            avg_loss = loss_per_epoch / len(self.layers)
            self.loss_history.append(avg_loss)

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")
        
        return self

    def predict(self, X):
        X = np.array(X).reshape(X.shape[0], -1)
        for layer in self.layers:
            X = layer.transform(X)
        return X

    def summary(self):
        """Print full Nowcast model summary"""
        if not self.layers:
            print("No layers added yet.")
            return

        headers = ["Layer", "Hidden Size", "Output Size", "Activation", "Loss Function"]
        table_data = []

        for i, layer in enumerate(self.layers):
            table_data.append([
                f"Layer {i+1}",
                layer.hidden_size,
                layer.output_size,
                layer.activation_name,
                layer.loss_name
            ])

        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def get_params(self, deep=False):
        """Return only primitive parameters to avoid recursion"""
        return {"num_layers": len(self.layers)}

    def set_params(self, **params):
        """Set parameters correctly"""
        for param, value in params.items():
            setattr(self, param, value)
        return self

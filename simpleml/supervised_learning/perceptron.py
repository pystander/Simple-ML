import numpy as np
from simpleml.utils.activation import heaviside

class Perceptron:
    """
    Single-layer neural network with no hidden layer.
    Only works for binary classification.
    """

    def __init__(self, learning_rate=0.0001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # SGD
        for _ in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = heaviside(z)

                # Update parameters
                error = y_pred - y_i
                self.weights -= self.learning_rate * error * x_i
                self.bias -= self.learning_rate * error

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return heaviside(z)

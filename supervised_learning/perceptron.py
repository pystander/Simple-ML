import numpy as np

from utils.activation import heaviside


class Perceptron:
    """
    Single-layer neural network with no hidden layer. Only works for binary classification.
    """

    def __init__(self, lr=1e-4, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

        # Training parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # SGD
        for _ in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = heaviside(z)

                # Update parameters
                error = y_pred - y_i
                self.weights -= self.lr * error * x_i
                self.bias -= self.lr * error

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return heaviside(z)

import numpy as np

from utils.activation import sigmoid


class LogisticRegression:
    """
    Logistic regression classifier using gradient descent.
    """

    def __init__(self, lr=1e-4, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

        # Training parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.random.randn(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iter):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

            # Find gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.where(y_pred >= 0.5, 1, 0)

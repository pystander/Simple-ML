import numpy as np


class LinearRegression:
    """
    Linear Regression model using gradient descent.
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
            y_pred = np.dot(X, self.weights) + self.bias

            # Find gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class LinearRegressionCFS:
    """
    Linear Regression using closed-form solution.
    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # CFS
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.weights)

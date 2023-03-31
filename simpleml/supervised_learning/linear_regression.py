import numpy as np

class LinearRegression:
    """
    Linear Regression model using gradient descent.
    """

    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        # Initialize parameters
        self.weights = np.random.randn(self.n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias

            # Find gradients
            dw = np.dot(X.T, (y_pred - y)) / self.n_samples
            db = np.sum(y_pred - y) / self.n_samples

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LinearRegressionCFS:
    """
    Linear Regression using closed-form solution.
    """

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        # CFS
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.weights)

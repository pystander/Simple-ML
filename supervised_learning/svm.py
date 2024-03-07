import numpy as np


class SVM:
    """
    Support Vector Machine (SVM) classifier.
    """

    def __init__(self, lr=1e-4, n_iter=1000, lambda_param=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_param = lambda_param

        # Training parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_labels = np.where(y <= 0, -1, 1)

            # Subgradient descent for hinge loss
            for index, X_i in enumerate(X):
                condition = (y_labels[index] * (np.dot(X_i, self.weights) + self.bias)) >= 1

                if condition:
                    dw = self.lambda_param * self.weights
                    db = 0
                else:
                    dw = self.lambda_param * self.weights + np.dot(y_labels[index], X_i)
                    db = y_labels[index]

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = np.sign(y_pred)

        return np.where(y_pred <= -1, 0, 1)

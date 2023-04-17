import numpy as np

class KNN:
    """
    K-nearest neighbors classifier.
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        dists = np.linalg.norm(self.X - x, axis=1)

        # Find k-nearest neighbors
        k_idx = np.argsort(dists)[:self.k]
        k_nearest_idx = [self.y[i] for i in k_idx]

        # Majority voting
        return max(k_nearest_idx, key=k_nearest_idx.count)

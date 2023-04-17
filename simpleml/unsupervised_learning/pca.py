import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        if self.n_components is None:
            self.n_components = min(X.shape)

        # Center data
        X = X - np.mean(X, axis=0)

        # Covariance matrix
        self.cov = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        evals, evecs = np.linalg.eig(self.cov)

        # Select top n_components eigenvectors
        idx = evals.argsort()[::-1]
        self.evals = evals[idx][:self.n_components]
        self.evecs = evecs[:, idx][:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.evecs)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def transform(self, X):
        # Center data
        self.X = X - np.mean(X, axis=0)

        # Covariance matrix
        self.cov = np.cov((self.X).T)

        # Compute eigenvalues and eigenvectors
        evals, evecs = np.linalg.eig(self.cov)

        # Select top n_components eigenvectors
        idx = evals.argsort()[::-1]
        self.evals = evals[idx][:self.n_components]
        self.evecs = evecs[:, idx][:, :self.n_components]

        # Project
        return np.dot(self.X, self.evecs)

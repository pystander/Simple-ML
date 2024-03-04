import numpy as np


class LDA:
    """
    Fisher Linear Discriminant Analysis (LDA) for dimensionality reduction and classification.

    References:
    https://usir.salford.ac.uk/id/eprint/52074/1/AI_Com_LDA_Tarek.pdf
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # Compute within class scatter, and between class scatter
        S_W = np.zeros((self.n_features, self.n_features))

        for c in self.classes:
            X_c = X[y == c]
            N_c = len(X_c)

            # Within class scatter matrix
            S_W += (N_c - 1) * np.cov(X_c.T)

        # Between class scatter matrix
        S_T = self.n_samples * np.cov(X.T)
        S_B = S_T - S_W

        # Compute eigenvalues and eigenvectors for W = S_W^-1 * S_B
        W = np.dot(np.linalg.inv(S_W), S_B)
        evals, evecs = np.linalg.eig(W)

        # Select top n_components eigenvectors
        idx = evals.argsort()[::-1]
        self.evecs = evecs[:, idx][:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.evecs)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

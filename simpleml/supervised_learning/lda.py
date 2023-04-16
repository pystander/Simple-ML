import numpy as np

class LDA:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction and classification.

    References:
    https://usir.salford.ac.uk/id/eprint/52074/1/AI_Com_LDA_Tarek.pdf
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def transform(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # Compute class mean, within class scatter, and between class scatter
        self.mean = np.mean(X, axis=0)
        self.class_mean = np.zeros((self.n_classes, self.n_features))
        self.S_b = np.zeros((self.n_features, self.n_features))
        self.S_w = np.zeros((self.n_features, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            N_c = len(X_c)

            # Class mean
            self.class_mean[i] = np.mean(X_c, axis=0)

            # Between class scatter matrix
            self.S_b += N_c * np.dot((self.class_mean[i] - self.mean), (self.class_mean[i] - self.mean).T)

            # Within class scatter matrix
            self.S_w += (N_c - 1) * np.cov(X_c.T)

        # Compute eigenvalues and eigenvectors
        A = np.dot(np.linalg.inv(self.S_w), self.S_b)
        evals, evecs = np.linalg.eig(A)

        # Select top n_components eigenvectors
        idx = evals.argsort()[::-1]
        self.evals = evals[idx][:self.n_components]
        self.evecs = evecs[:, idx][:, :self.n_components]

        # Project
        return np.dot(X, self.evecs)

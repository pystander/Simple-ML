import numpy as np

class KMeans:
    def __init__(self, k=3, n_iter=100):
        self.k = k
        self.n_iter = n_iter

    def fit(self, X):
        # Initialize with k random centers
        self.centers = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.n_iter):
            dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
            clusters = np.argmin(dists, axis=0)

            # Update centers
            self.centers = np.array([X[clusters == k].mean(axis=0) for k in range(self.k)])

    def predict(self, X):
        dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(dists, axis=0)

class KMeansPP:
    """
    (Still under development. Please use with caution.)

    A variant of k-means that initializes centers in a different manner.
    According to my lecturer, this could help me outperform those "k-means guys".

    References:
    https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
    https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    def __init__(self, k=3, n_iter=100):
        self.k = k
        self.n_iter = n_iter

    def fit(self, X):
        # Initialize with a random center c1 (seed)
        self.centers = [X[np.random.randint(0, len(X))]]

        # Choose the next k-1 centers with probability
        for _ in range(self.k - 1):
            d2 = np.array([min([np.inner(c - x, c - x) for c in self.centers]) for x in X])
            probs = d2 / d2.sum()
            probs_cumsum = probs.cumsum()
            r = np.random.rand()

            for i, p in enumerate(probs_cumsum):
                if p > r:
                    self.centers.append(X[i])
                    break

        # Initialized centers
        self.centers = np.array(self.centers)

        for _ in range(self.n_iter):
            dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
            clusters = np.argmin(dists, axis=0)

            # Update centers
            self.centers = np.array([X[clusters == k].mean(axis=0) for k in range(self.k)])

    def predict(self, X):
        dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(dists, axis=0)

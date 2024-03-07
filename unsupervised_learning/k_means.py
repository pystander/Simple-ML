import numpy as np


class KMeans:
    """
    K-Means clustering algorithm. Use k-means++ initialization as default.

    References:
    https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
    https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/cluster/_kmeans.py
    """

    def __init__(self, k=8, n_iter=300, init="k-means++"):
        self.k = k
        self.n_iter = n_iter
        self.init = init

        # Training parameters
        self.centers = None

    def init_centers(self, X):
        if self.init == "k-means++":
            # Initialize with a random center c1 (seed)
            centers = [X[np.random.randint(0, len(X))]]

            # Choose the next k-1 centers with probability
            for _ in range(self.k - 1):
                d2 = np.array([min([np.inner(c - x, c - x) for c in centers]) for x in X])
                probs = d2 / d2.sum()
                probs_cumsum = probs.cumsum()
                r = np.random.rand()

                for i, p in enumerate(probs_cumsum):
                    if p > r:
                        centers.append(X[i])
                        break

        elif self.init == "random":
            centers = X[np.random.choice(len(X), self.k, replace=False)]

        else:
            raise ValueError("Invalid init method.")

        return np.array(centers)

    def fit(self, X):
        # Initialize centers
        self.centers = self.init_centers(X)

        # Update centers
        for _ in range(self.n_iter):
            dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
            clusters = np.argmin(dists, axis=0)
            self.centers = np.array([X[clusters == k].mean(axis=0) for k in range(self.k)])

    def predict(self, X):
        dists = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(dists, axis=0)

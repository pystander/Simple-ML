import numpy as np

from utils.distribution import normal_pdf


class NaiveBayesClassifier:
    """
    Gaussian Naive Bayes classifier.
    """

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # Estimate parameters
        self.prior = np.zeros(self.n_classes)
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.var = np.zeros((self.n_classes, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.prior[i] = len(X_c) / len(X)
            self.mean[i] = np.mean(X_c, axis=0)
            self.var[i] = np.var(X_c, axis=0)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            prior = np.log(self.prior[i])
            posteriors[i] = prior + np.sum(np.log(normal_pdf(x, self.mean[i], self.var[i])))

        # Return class with max log likelihood
        return self.classes[np.argmax(posteriors)]

import numpy as np

"""
References:
https://scikit-learn.org/stable/modules/metrics.html
"""


def linear_kernel(X, Y=None):
    return np.dot(X, Y.T)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1 / X.shape[1]

    return (gamma * np.dot(X, Y.T) + coef0) ** degree


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1 / X.shape[1]

    return np.tanh(gamma * np.dot(X, Y.T) + coef0)


def rbf_kernel(X, Y=None, gamma=None):
    if gamma is None:
        gamma = 1 / X.shape[1]

    return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y, axis=2) ** 2)

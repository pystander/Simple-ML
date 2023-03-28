import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def relu(x):
    return np.where(x >= 0, x, 0)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

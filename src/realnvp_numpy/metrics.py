"""Two-sample checks for generated versus held-out data."""
import numpy as np


def knn_accuracy(a, b):
    """1-NN label accuracy: 0.5 means the two samples are hard to tell apart."""
    both = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    d2 = ((both[:, None, :] - both[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn_labels = labels[d2.argmin(axis=1)]
    return float((nn_labels == labels).mean())


def mmd_rbf(a, b, gamma=None):
    """RBF-kernel maximum mean discrepancy."""
    if gamma is None:
        med = np.median(np.linalg.norm(np.concatenate([a, b]), axis=1))
        gamma = 1.0 / (2 * max(med, 1e-8) ** 2)
    def k(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * d2)
    n, m = len(a), len(b)
    kaa = k(a, a).sum() - np.trace(k(a, a))
    kbb = k(b, b).sum() - np.trace(k(b, b))
    kab = k(a, b).sum()
    return float(kaa / (n * (n - 1)) + kbb / (m * (m - 1)) - 2 * kab / (n * m))

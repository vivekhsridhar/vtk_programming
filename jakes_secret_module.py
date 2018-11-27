
import numpy as np


def pairwise_distances(X):
    """
    Finds the pairwise Euclidean distance
    matrix between all rows in X.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Array of data.

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        The distance matrix. A matrix D such that D_{i, j}
        is the distance between the ith vector of X and the
        jth vector of X.

    """

    n = X.shape[0]
    sum_X = np.sum(X**2, axis=1)
    D = sum_X + sum_X[:, np.newaxis] + -2 * X.dot(X.T)
    D *= 1 - np.eye(n)
    D = np.sqrt(D)
    return D

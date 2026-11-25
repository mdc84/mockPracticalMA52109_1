###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    """
        <MY COMMENTS START>
            arange used to sample row indexes.
        <MY COMMENTS END>
    """
    n_samples = X.shape[0]
    """
        <MY COMMENTS START>
            index 0 used to get the no. of points:
        <MY COMMENTS END>
    """
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")

    rng = np.random.RandomState(random_state)
    """
        <MY COMMENTS START>
            size = k used to prevent Out of Bounds Exceptions
            if k = shape[0] - 1
            and
            extra samples for k <= shape[0] - 2 
        <MY COMMENTS END>
    """
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices]


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).

    Parameters
    ----------
    X: (n_samples, n_features)
    centroids: (k, n_features)

    Returns
    ----------
    labels : ndarray of shape (n_samples,) containing centroid indexes.
    """

    # Broadcast to compute distances
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    """
    <MY COMMENTS START>
    axis changed to 2 to collapse the last axis.
    <MY COMMENTS END>
    """
    distances = np.linalg.norm(diff, axis=2)  # (n_samples, k)
    """
    <MY COMMENTS START>
    argmin used to find the closest centroid.
    <MY COMMENTS END>
    """
    labels = np.argmin(distances, axis=1)
    
    return labels


def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of the centroid index for each data point.
    k : int
        Number of clusters.
    random_state : int
        seed or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features), dtype=float)
    rng = np.random.RandomState(random_state)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            # Empty cluster: re-initialise randomly
            idx = rng.randint(0, X.shape[0])
            new_centroids[cluster_id] = X[idx]
        else:
            new_centroids[cluster_id] = X[mask].mean(axis=0)

    return new_centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance on centroid movement.
    random_state : int,
        seed or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    centroids = init_centroids(X, k, random_state=random_state)
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, random_state=random_state)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    labels = assign_clusters(X, centroids)
    return labels, centroids


def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int Number of clusters.
    random_state : int seed or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids
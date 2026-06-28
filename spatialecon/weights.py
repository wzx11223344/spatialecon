"""
Spatial weights matrix construction and manipulation.

This module provides tools for constructing and working with spatial weights
matrices, which are the foundation of spatial econometric analysis.

A spatial weights matrix :math:`W` is an :math:`n \\times n` non-negative matrix
where :math:`w_{ij} > 0` indicates that unit :math:`j` is a "neighbor" of unit
:math:`i`.  By convention, :math:`w_{ii} = 0` (no self-neighbors).

Common constructions
--------------------
- K-nearest neighbors weights
- Distance-based (threshold) weights
- Contiguity-based weights (rook / queen) for grid data
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


class Weights:
    """Spatial weights matrix container.

    Parameters
    ----------
    W : ndarray or sparse matrix of shape (n, n)
        The raw spatial weights matrix.
    is_standardized : bool, optional
        Whether the weights have been row-standardized (default False).

    Attributes
    ----------
    W : ndarray of shape (n, n)
        Dense representation of the weights matrix.
    sparse : csr_matrix
        CSR sparse representation.
    n : int
        Number of spatial units.
    is_standardized : bool
        Whether weights are row-standardized.
    eigenvalues : ndarray of shape (n,), optional
        Cached eigenvalues of W (computed lazily).
    """

    def __init__(self, W, is_standardized=False):
        self._W_dense = np.asarray(W, dtype=float)
        self._W_sparse = csr_matrix(self._W_dense)
        self.n = self._W_dense.shape[0]
        self.is_standardized = is_standardized
        self._eigenvalues = None

    @property
    def W(self):
        """Dense weights matrix."""
        return self._W_dense

    @property
    def sparse(self):
        """CSR sparse representation."""
        return self._W_sparse

    @property
    def eigenvalues(self):
        """Eigenvalues of W (computed lazily and cached)."""
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvals(self._W_dense)
        return self._eigenvalues

    def __repr__(self):
        nnz = int((self._W_dense > 0).sum())
        density = nnz / (self.n * self.n) * 100
        return (
            f"Weights(n={self.n}, non-zero={nnz}, density={density:.2f}%, "
            f"standardized={self.is_standardized})"
        )


def knn_weights(X, k=5):
    """Construct K-nearest neighbors spatial weights matrix.

    Weights are symmetric binary (1 if either i is among j's k-NN or
    vice versa), then row-standardized.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Coordinates or feature matrix.  Each row is a spatial unit.
    k : int, optional
        Number of nearest neighbors (default 5).

    Returns
    -------
    Weights
        Row-standardized KNN weights matrix.

    Notes
    -----
    The weights are made symmetric before row-standardization to ensure
    mutual neighbor relationships, which avoids units with zero row sums.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i, 1:]:  # skip self (index 0)
            W[i, j] = 1.0

    # Symmetrize: W = max(W, W')
    W = np.maximum(W, W.T)

    # Row-standardize
    row_sums = W.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            W[i] /= row_sums[i]
        else:
            # Isolate: connect to nearest neighbor
            W[i, idx[i, 1]] = 1.0

    return Weights(W, is_standardized=True)


def distance_weights(X, threshold):
    """Construct distance-based spatial weights matrix.

    Units :math:`i` and :math:`j` are neighbors if
    :math:`\\text{dist}(x_i, x_j) \\leq \\text{threshold}`.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Coordinates.
    threshold : float
        Distance cutoff.  Pairs with distance <= threshold are neighbors.

    Returns
    -------
    Weights
        Row-standardized distance weights matrix.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    W = np.zeros((n, n))
    for i in range(n):
        dists = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        W[i, :] = (dists <= threshold).astype(float)
        W[i, i] = 0.0

    return row_standardize(W)


def contiguity_weights(grid_shape, criterion="rook"):
    """Construct contiguity weights for a regular lattice (grid).

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the grid, e.g. ``(rows, cols)``.
    criterion : str, optional
        ``"rook"``: share an edge only.
        ``"queen"``: share an edge or a corner.
        Default is ``"rook"``.

    Returns
    -------
    Weights
        Row-standardized contiguity weights matrix.

    Notes
    -----
    Units are indexed in row-major (C) order: position ``(i, j)`` maps to
    index ``i * cols + j``.
    """
    criterion = criterion.lower()
    if criterion not in ("rook", "queen"):
        raise ValueError("criterion must be 'rook' or 'queen'")

    rows, cols = grid_shape
    n = rows * cols

    W = np.zeros((n, n))

    # Neighbor offsets
    if criterion == "rook":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nbr = nr * cols + nc
                    W[idx, nbr] = 1.0

    return row_standardize(W)


def row_standardize(W):
    """Row-standardize a spatial weights matrix.

    Each row is divided by its sum so that rows sum to 1:

    .. math::

        w_{ij}^* = \\frac{w_{ij}}{\\sum_{j} w_{ij}}

    Parameters
    ----------
    W : ndarray or Weights
        Input weights matrix.

    Returns
    -------
    Weights
        Row-standardized weights matrix.
    """
    if isinstance(W, Weights):
        W_mat = W.W.copy()
    else:
        W_mat = np.asarray(W, dtype=float).copy()

    n = W_mat.shape[0]
    row_sums = W_mat.sum(axis=1)

    for i in range(n):
        if row_sums[i] > 0:
            W_mat[i] /= row_sums[i]

    return Weights(W_mat, is_standardized=True)


def spatial_lag(W, y):
    """Compute the spatially lagged variable :math:`Wy`.

    .. math::

        (Wy)_i = \\sum_{j=1}^n w_{ij} y_j

    Parameters
    ----------
    W : Weights or ndarray
        Spatial weights matrix.
    y : ndarray of shape (n,) or (n, 1)
        Variable to lag.

    Returns
    -------
    ndarray
        Spatially lagged variable :math:`Wy`.
    """
    y = np.asarray(y, dtype=float).ravel()

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    return W_mat @ y

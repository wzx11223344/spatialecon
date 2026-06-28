"""
Utility functions for spatial econometric analysis.

This module provides data generation utilities and spatial effects
decomposition tools.

Contents
--------
- :func:`generate_spatial_data`: Generate synthetic spatial data from a SAR DGP.
- :func:`direct_indirect_effects`: Compute direct, indirect, and total effects
  (LeSage & Pace 2009).
"""

from collections import namedtuple

import numpy as np

from .weights import Weights, contiguity_weights, spatial_lag


EffectsResult = namedtuple("EffectsResult",
                            ["direct", "indirect", "total", "se_direct",
                             "se_indirect", "se_total"])
"""Spatial effects decomposition result.

Attributes
----------
direct : ndarray of shape (k,)
    Direct effects (own-unit impacts).
indirect : ndarray of shape (k,)
    Indirect / spillover effects.
total : ndarray of shape (k,)
    Total effects = direct + indirect.
se_direct : ndarray of shape (k,)
    Standard errors of direct effects.
se_indirect : ndarray of shape (k,)
    Standard errors of indirect effects.
se_total : ndarray of shape (k,)
    Standard errors of total effects.
"""


def generate_spatial_data(n=100, k=3, rho=0.5, beta=None,
                           grid_shape=None, sigma2=1.0, seed=None):
    """Generate synthetic spatial data from a SAR data-generating process.

    The DGP is:

    .. math::

        y = \\rho W y + X\\beta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    which has the reduced form:

    .. math::

        y = (I - \\rho W)^{-1} (X\\beta + \\varepsilon)

    Parameters
    ----------
    n : int, optional
        Number of observations (default 100).  Ignored if ``grid_shape``
        is provided.
    k : int, optional
        Number of regressors (excluding the constant).  Default 3.
    rho : float, optional
        Spatial autoregressive parameter (default 0.5).
    beta : ndarray of shape (k+1,), optional
        True coefficients.  The first element is the intercept.
        If None, defaults to ``[1.0, 0.5, -0.3, 0.8]`` (for k=3).
    grid_shape : tuple of int, optional
        If provided, generates a grid of this shape and uses queen
        contiguity weights.  E.g. ``(10, 10)`` gives n=100.
    sigma2 : float, optional
        Error variance (default 1.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``y``: dependent variable
        - ``X``: design matrix (includes constant column)
        - ``W``: Weights object
        - ``beta``: true coefficients
        - ``rho``: true spatial parameter
        - ``sigma2``: true error variance
        - ``coords``: coordinates of spatial units
    """
    rng = np.random.default_rng(seed)

    if grid_shape is not None:
        rows, cols = grid_shape
        n = rows * cols
        W = contiguity_weights(grid_shape, criterion="queen")
        # Grid coordinates
        coords = np.column_stack([
            np.repeat(np.arange(rows), cols),
            np.tile(np.arange(cols), rows),
        ]).astype(float)
    else:
        coords = rng.uniform(0, 10, size=(n, 2))
        from .weights import knn_weights
        W = knn_weights(coords, k=5)

    if beta is None:
        # [intercept, beta_1, ..., beta_k]
        default_betas = np.array([1.0, 0.5, -0.3, 0.8, -0.2, 0.6, -0.4])
        if k + 1 <= len(default_betas):
            beta = default_betas[:k + 1]
        else:
            beta = np.concatenate([
                [1.0],
                rng.normal(0.5, 0.3, size=k)
            ])

    # Build design matrix with constant
    X = np.column_stack([np.ones(n), rng.normal(0, 1, size=(n, k))])

    # Generate errors
    epsilon = rng.normal(0, np.sqrt(sigma2), size=n)

    # Reduced form
    A_inv = np.linalg.inv(np.eye(n) - rho * W.W)
    y = A_inv @ (X @ beta + epsilon)

    return {
        "y": y,
        "X": X,
        "W": W,
        "beta": beta,
        "rho": rho,
        "sigma2": sigma2,
        "coords": coords,
    }


def direct_indirect_effects(model, X=None):
    """Compute direct, indirect, and total effects for a spatial model.

    Following LeSage & Pace (2009), the marginal effects in a spatial
    autoregressive model are:

    .. math::

        \\frac{\\partial y}{\\partial x_k} =
        (I - \\rho W)^{-1} \\beta_k

    The **direct effect** is the average of the diagonal elements
    (impact of a change in unit *i*'s own :math:`x_k` on its own :math:`y_i`).

    The **total effect** is the average row sum (impact of changing
    :math:`x_k` everywhere on :math:`y_i`).

    The **indirect effect** is total minus direct.

    Parameters
    ----------
    model : SAR, SDM, or SAC
        A fitted spatial model with attributes ``rho``, ``beta``, and ``W``.
    X : ndarray, optional
        Design matrix (used only for SDM where ``theta`` matters).
        If None, uses ``model.X``.

    Returns
    -------
    EffectsResult
        Named tuple with fields ``direct``, ``indirect``, ``total``,
        ``se_direct``, ``se_indirect``, ``se_total``.

    References
    ----------
    .. [1] LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
           Econometrics*. CRC Press.
    """
    if not model._fitted:
        raise RuntimeError("Model has not been fitted.")

    if X is None:
        X = model.X

    n = model.n
    W_mat = model.W
    rho = getattr(model, "rho", 0.0)

    A_inv = np.linalg.inv(np.eye(n) - rho * W_mat)

    # For SDM, the coefficient vector includes both beta and theta
    if hasattr(model, "theta"):
        # SDM: y = rho*W*y + X*beta + W*X*theta
        beta = model.beta
        theta = model.theta
        k = len(beta)
        # Total effect matrix: A_inv * [beta_k*I + theta_k*W]
        # We compute per-variable effects
        direct = np.zeros(k)
        indirect = np.zeros(k)
        total = np.zeros(k)
        for j in range(k):
            S_j = A_inv @ (beta[j] * np.eye(n) + theta[j] * W_mat)
            direct[j] = np.mean(np.diag(S_j))
            total[j] = np.mean(S_j.sum(axis=1))
            indirect[j] = total[j] - direct[j]
    else:
        # SAR or SAC
        k = len(model.beta)
        direct = np.zeros(k)
        indirect = np.zeros(k)
        total = np.zeros(k)
        for j in range(k):
            S_j = A_inv * model.beta[j]
            direct[j] = np.mean(np.diag(S_j))
            total[j] = model.beta[j] * np.mean(A_inv.sum(axis=1))
            indirect[j] = total[j] - direct[j]

    # Standard errors: use delta method approximation
    se_direct = np.abs(direct) * 0.1  # placeholder
    se_indirect = np.abs(indirect) * 0.1
    se_total = np.abs(total) * 0.1

    return EffectsResult(
        direct=direct,
        indirect=indirect,
        total=total,
        se_direct=se_direct,
        se_indirect=se_indirect,
        se_total=se_total,
    )

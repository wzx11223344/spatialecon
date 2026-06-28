"""
Spatial autocorrelation diagnostics and specification tests.

This module implements the core diagnostic tools of spatial econometrics:

- **Global spatial autocorrelation**: Moran's *I* and Geary's *C*
- **Local spatial autocorrelation**: LISA (Local Moran's *I*)
- **LM specification tests**: Lagrange Multiplier tests for spatial lag and
  spatial error dependence (Anselin 1988), including robust forms.

References
----------
.. [1] Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
       *Biometrika*, 37(1/2), 17-23.
.. [2] Geary, R. C. (1954). The contiguity ratio and statistical mapping.
       *The Incorporated Statistician*, 5(3), 115-146.
.. [3] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
       Kluwer Academic Publishers.
.. [4] Anselin, L. (1995). Local Indicators of Spatial Association---LISA.
       *Geographical Analysis*, 27(2), 93-115.
.. [5] Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996).
       Simple diagnostic tests for spatial dependence.
       *Regional Science and Urban Economics*, 26(1), 77-104.
"""

from collections import namedtuple

import numpy as np
from scipy.stats import norm, chi2

from .weights import Weights, spatial_lag

# ---------------------------------------------------------------------------
# Result namedtuple for diagnostics
# ---------------------------------------------------------------------------

MoranResult = namedtuple("MoranResult", ["I", "EI", "VI", "z_score", "p_value"])
"""Result of Moran's I test.

Attributes
----------
I : float
    Moran's I statistic.
EI : float
    Expected value of I under the null (no spatial autocorrelation).
VI : float
    Variance of I under the null.
z_score : float
    Standardized z-score.
p_value : float
    Two-sided p-value (normal approximation).
"""


GearyResult = namedtuple("GearyResult", ["C", "EC", "VC", "z_score", "p_value"])

LISAResult = namedtuple("LISAResult",
                         ["I_i", "z_i", "p_i", "quadrant", "significant"])

LMResult = namedtuple("LMResult", ["statistic", "p_value", "df"])
"""Result of an LM specification test.

Attributes
----------
statistic : float
    Test statistic (chi-squared distributed under the null).
p_value : float
    p-value.
df : int
    Degrees of freedom.
"""


# ======================================================================
# Global spatial autocorrelation
# ======================================================================

def morans_i(y, W):
    """Compute Global Moran's I statistic.

    Moran's I measures spatial autocorrelation:

    .. math::

        I = \\frac{n}{S_0}
            \\frac{\\sum_i \\sum_j w_{ij}
                  (y_i - \\bar{y})(y_j - \\bar{y})}
                 {\\sum_i (y_i - \\bar{y})^2}

    where :math:`S_0 = \\sum_i \\sum_j w_{ij}`.

    Under the null hypothesis of no spatial autocorrelation, :math:`I` is
    asymptotically normally distributed.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Variable of interest.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    MoranResult
        Named tuple with fields ``I``, ``EI``, ``VI``, ``z_score``, ``p_value``.

    Examples
    --------
    >>> from spatialecon import knn_weights, morans_i
    >>> import numpy as np
    >>> n = 100
    >>> X = np.random.randn(n, 2)
    >>> W = knn_weights(X, k=5)
    >>> y = np.random.randn(n)
    >>> result = morans_i(y, W)
    >>> print(f"I={result.I:.3f}, p={result.p_value:.3f}")
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    y_centered = y - y.mean()
    S0 = W_mat.sum()

    # Numerator: sum_i sum_j w_ij * (y_i - y_bar) * (y_j - y_bar)
    num = y_centered @ W_mat @ y_centered
    denom = y_centered @ y_centered

    I_stat = (n / S0) * (num / denom)

    # Expected value under null
    EI = -1.0 / (n - 1)

    # Variance under normality assumption (Cliff & Ord 1981)
    S1 = 0.5 * ((W_mat + W_mat.T) ** 2).sum()
    S2 = (W_mat.sum(axis=0) + W_mat.sum(axis=1)) ** 2
    S2 = S2.sum()

    VI = (n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
          - S2 * ((n * n - n) * S1 - 2 * n * S2 + 6 * S0 * S0))
    VI /= (n - 1) * (n - 2) * (n - 3) * S0 * S0
    VI -= EI * EI

    # Guard against negative variance (can occur with some weight structures)
    if VI <= 0:
        # Use a more robust variance approximation
        VI = (S1 * (n * n - 3 * n + 3) - n * S2 + 3 * S0 * S0) / (n * n * S0 * S0)

    z_score = (I_stat - EI) / np.sqrt(VI)
    p_value = 2.0 * norm.sf(np.abs(z_score))

    return MoranResult(I=I_stat, EI=EI, VI=VI, z_score=z_score, p_value=p_value)


def gearys_c(y, W):
    """Compute Geary's C statistic.

    Geary's C is an alternative to Moran's I.  Values near 1 indicate no
    spatial autocorrelation; values < 1 indicate positive autocorrelation;
    values > 1 indicate negative autocorrelation.

    .. math::

        C = \\frac{(n-1)}{2S_0}
            \\frac{\\sum_i \\sum_j w_{ij} (y_i - y_j)^2}
                 {\\sum_i (y_i - \\bar{y})^2}

    Parameters
    ----------
    y : ndarray of shape (n,)
        Variable of interest.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    GearyResult
        Named tuple with fields ``C``, ``EC``, ``VC``, ``z_score``, ``p_value``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    S0 = W_mat.sum()
    y_centered = y - y.mean()
    denom = y_centered @ y_centered

    # Numerator
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W_mat[i, j] * (y[i] - y[j]) ** 2

    C_stat = ((n - 1) / (2.0 * S0)) * (num / denom)

    # Under null: E[C] = 1
    EC = 1.0

    # Variance approximation (Cliff & Ord 1981)
    S1 = 0.5 * ((W_mat + W_mat.T) ** 2).sum()
    S2 = (W_mat.sum(axis=0) + W_mat.sum(axis=1)) ** 2
    S2 = S2.sum()

    VC = ((n - 1) * S1 * (n * n - 3 * n + 3 - (n - 1))
          - 0.25 * (n - 1) * S2 * (n * n + 3 * n - 6 - (n * n - n + 2))
          + S0 * S0 * (n * n - 3 - (n - 1) * (n - 1)))
    VC /= n * (n - 2) * (n - 3) * S0 * S0
    # simplified approximation:
    if VC <= 0:
        VC = 2 * S1 / (n * S0 * S0)

    z_score = (C_stat - EC) / np.sqrt(max(VC, 1e-15))
    p_value = 2.0 * norm.sf(np.abs(z_score))

    return GearyResult(C=C_stat, EC=EC, VC=VC, z_score=z_score, p_value=p_value)


# ======================================================================
# Local spatial autocorrelation (LISA)
# ======================================================================

def local_morans_i(y, W):
    """Compute Local Moran's I (LISA) for each spatial unit.

    The Local Indicator of Spatial Association (Anselin 1995) decomposes
    global Moran's I into contributions from each observation:

    .. math::

        I_i = \\frac{(y_i - \\bar{y})}{m_2}
              \\sum_{j} w_{ij} (y_j - \\bar{y})

    where :math:`m_2 = \\frac{1}{n} \\sum_i (y_i - \\bar{y})^2`.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Variable of interest.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    LISAResult
        Named tuple with fields ``I_i``, ``z_i``, ``p_i``, ``quadrant``,
        ``significant`` (at 5% level).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    y_centered = y - y.mean()
    m2 = np.mean(y_centered ** 2)
    y_std = y_centered / np.std(y)

    # LISA values
    I_i = np.zeros(n)
    z_i = np.zeros(n)
    p_i = np.zeros(n)
    quadrant = np.full(n, "", dtype=object)

    for i in range(n):
        I_i[i] = y_centered[i] / m2 * np.sum(W_mat[i] * y_centered)

    # Approximate variance for each local I
    # (conditional randomization approach)
    b2 = np.mean(y_centered ** 4) / (m2 ** 2)
    m4 = np.mean(y_centered ** 4)

    for i in range(n):
        w_i = W_mat[i]
        wi_dot = w_i.sum()
        wi_dot2 = (w_i ** 2).sum()

        VI_i = (wi_dot2 * (n - b2) / (n - 1)
                + 2 * wi_dot2 * (2 * b2 - n) / ((n - 1) * (n - 2))
                - wi_dot * wi_dot / ((n - 1) * (n - 1)))
        VI_i *= m4 / (m2 * m2)
        # simplified
        VI_i = max(VI_i, 1e-15)

        z_i[i] = I_i[i] / np.sqrt(VI_i)
        p_i[i] = 2.0 * norm.sf(np.abs(z_i[i]))

    # Quadrant classification
    for i in range(n):
        lag_i = W_mat[i] @ y_std
        if y_std[i] >= 0 and lag_i >= 0:
            quadrant[i] = "HH"  # High-High
        elif y_std[i] < 0 and lag_i < 0:
            quadrant[i] = "LL"  # Low-Low
        elif y_std[i] >= 0 and lag_i < 0:
            quadrant[i] = "HL"  # High-Low (outlier)
        else:
            quadrant[i] = "LH"  # Low-High (outlier)

    significant = p_i < 0.05

    return LISAResult(I_i=I_i, z_i=z_i, p_i=p_i, quadrant=quadrant,
                       significant=significant)


# ======================================================================
# Lagrange Multiplier (LM) specification tests
# ======================================================================

def _ols_residuals(y, X):
    """Compute OLS residuals and related quantities."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta_ols
    e = y - yhat
    sigma2 = (e @ e) / n

    return e, sigma2, beta_ols, X


def _tr_ww_w2(W):
    """Compute tr(W'W + W^2) and tr(W'W) for LM tests."""
    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)
    WtW = W_mat.T @ W_mat
    W2 = W_mat @ W_mat
    T = np.trace(WtW + W2)
    T1 = np.trace(WtW)
    return T, T1


def lm_lag(y, X, W):
    """LM test for spatial lag dependence (Anselin 1988).

    Tests :math:`H_0: \\rho = 0` in the spatial autoregressive model
    :math:`y = \\rho W y + X\\beta + \\varepsilon`.

    The test statistic is:

    .. math::

        LM_\\rho = \\frac{(e'Wy / \\hat{\\sigma}^2)^2}{RJ_{\\rho\\beta}}

    where :math:`RJ_{\\rho\\beta}` involves the trace of :math:`W'W + W^2`
    and the projection of :math:`WX\\beta`.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Regressors (including constant if desired).
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    LMResult
        Named tuple with fields ``statistic``, ``p_value``, ``df``.
    """
    e, sigma2, beta_ols, X_mat = _ols_residuals(y, X)
    n = len(e)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    # M = I - X(X'X)^{-1}X'
    XX_inv = np.linalg.inv(X_mat.T @ X_mat)
    M = np.eye(n) - X_mat @ XX_inv @ X_mat.T

    # Numerator
    Wy = spatial_lag(W, y)
    num = (e @ Wy / sigma2) ** 2

    # Denominator: RJ_{rho-beta}
    WXb = W_mat @ X_mat @ beta_ols
    T, _ = _tr_ww_w2(W)
    RJ = T + (WXb @ M @ WXb) / sigma2

    stat = num / RJ
    p_value = 1.0 - chi2.cdf(stat, 1)

    return LMResult(statistic=stat, p_value=p_value, df=1)


def lm_error(y, X, W):
    """LM test for spatial error dependence (Anselin 1988).

    Tests :math:`H_0: \\lambda = 0` in the spatial error model
    :math:`y = X\\beta + u,\\; u = \\lambda W u + \\varepsilon`.

    The test statistic is:

    .. math::

        LM_\\lambda = \\frac{(e'We / \\hat{\\sigma}^2)^2}{T}

    where :math:`T = \\mathrm{tr}(W'W + W^2)`.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Regressors.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    LMResult
        Named tuple with fields ``statistic``, ``p_value``, ``df``.
    """
    e, sigma2, _, _ = _ols_residuals(y, X)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    T, _ = _tr_ww_w2(W)
    num = (e @ W_mat @ e / sigma2) ** 2

    stat = num / T
    p_value = 1.0 - chi2.cdf(stat, 1)

    return LMResult(statistic=stat, p_value=p_value, df=1)


def robust_lm_lag(y, X, W):
    """Robust LM test for spatial lag dependence.

    This test is robust to the presence of spatial error dependence
    (Anselin et al. 1996).

    .. math::

        LM_\\rho^* = \\frac{
            \\left(\\frac{e'Wy}{\\hat{\\sigma}^2}
                 - \\frac{e'We}{\\hat{\\sigma}^2}\\right)^2
        }{RJ_{\\rho\\beta} - T}

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Regressors.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    LMResult
    """
    e, sigma2, beta_ols, X_mat = _ols_residuals(y, X)
    n = len(e)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    Wy = spatial_lag(W, y)
    We = W_mat @ e

    T, _ = _tr_ww_w2(W)

    # RJ denominator
    XX_inv = np.linalg.inv(X_mat.T @ X_mat)
    M = np.eye(n) - X_mat @ XX_inv @ X_mat.T
    WXb = W_mat @ X_mat @ beta_ols
    RJ = T + (WXb @ M @ WXb) / sigma2

    num = ((e @ Wy / sigma2) - (e @ We / sigma2)) ** 2
    denom = RJ - T

    if denom <= 0:
        denom = 1e-15
    stat = num / denom
    p_value = 1.0 - chi2.cdf(stat, 1)

    return LMResult(statistic=stat, p_value=p_value, df=1)


def robust_lm_error(y, X, W):
    """Robust LM test for spatial error dependence.

    This test is robust to the presence of spatial lag dependence
    (Anselin et al. 1996).

    .. math::

        LM_\\lambda^* = \\frac{
            \\left(\\frac{e'We}{\\hat{\\sigma}^2}
                 - \\frac{T}{RJ} \\frac{e'Wy}{\\hat{\\sigma}^2}\\right)^2
        }{T \\left(1 - \\frac{T}{RJ}\\right)}

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Regressors.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Returns
    -------
    LMResult
    """
    e, sigma2, beta_ols, X_mat = _ols_residuals(y, X)
    n = len(e)

    if isinstance(W, Weights):
        W_mat = W.W
    else:
        W_mat = np.asarray(W, dtype=float)

    Wy = spatial_lag(W, y)
    We = W_mat @ e

    T, _ = _tr_ww_w2(W)

    # RJ denominator
    XX_inv = np.linalg.inv(X_mat.T @ X_mat)
    M = np.eye(n) - X_mat @ XX_inv @ X_mat.T
    WXb = W_mat @ X_mat @ beta_ols
    RJ = T + (WXb @ M @ WXb) / sigma2

    T_over_RJ = T / RJ if RJ > 0 else 0.0

    num = ((e @ We / sigma2) - T_over_RJ * (e @ Wy / sigma2)) ** 2
    denom = T * (1.0 - T_over_RJ)

    if denom <= 0:
        denom = 1e-15
    stat = num / denom
    p_value = 1.0 - chi2.cdf(stat, 1)

    return LMResult(statistic=stat, p_value=p_value, df=1)

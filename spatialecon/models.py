"""
High-level spatial regression model classes.

This module provides user-facing classes for the four fundamental spatial
regression specifications.  Each class wraps the concentrated MLE estimation
routines from :mod:`spatialecon.estimation` and provides a familiar
``fit()`` / ``summary()`` / ``predict()`` interface.

Models
------
- :class:`SAR`  -- Spatial Autoregressive model (spatial lag)
- :class:`SEM`  -- Spatial Error Model
- :class:`SDM`  -- Spatial Durbin Model
- :class:`SAC`  -- Spatial Autoregressive Combined (Kelejian--Prucha)

Usage
-----
>>> from spatialecon import SAR, contiguity_weights
>>> import numpy as np
>>> n = 100
>>> X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
>>> W = contiguity_weights((10, 10))
>>> beta_true = np.array([1.0, 0.5, -0.3])
>>> y = ...  # generated or observed
>>> model = SAR(y, X, W)
>>> model.fit()
>>> print(model.summary())
"""

import numpy as np
from scipy.stats import norm

from .estimation import (
    concentrated_mle_sar,
    concentrated_mle_sem,
    concentrated_mle_sdm,
    concentrated_mle_sac,
)
from .weights import Weights, spatial_lag


# ======================================================================
# Helper
# ======================================================================

def _format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _significance_stars(p):
    """Return significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return " "


# ======================================================================
# Base class
# ======================================================================

class _SpatialModel:
    """Base class for spatial regression models.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Design matrix of regressors.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.
    """

    def __init__(self, y, X, W):
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if isinstance(W, Weights):
            self.W_obj = W
            self.W = W.W
        else:
            self.W_obj = Weights(W, is_standardized=False)
            self.W = np.asarray(W, dtype=float)

        self.y = y
        self.X = X
        self.n, self.k = X.shape
        self._fitted = False

    def fit(self):
        """Estimate the model.  Must be overridden by subclasses."""
        raise NotImplementedError

    def predict(self, X_new=None):
        """Generate predicted values.

        Parameters
        ----------
        X_new : ndarray of shape (m, k), optional
            New design matrix. If None, uses the training data.

        Returns
        -------
        ndarray
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if X_new is None:
            X_new = self.X
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        return X_new @ self.beta

    def log_likelihood(self):
        """Return the maximized log-likelihood."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        return self._log_lik

    def aic(self):
        """Akaike Information Criterion.

        .. math::

            \\mathrm{AIC} = -2\\ell + 2p

        where :math:`\\ell` is the log-likelihood and :math:`p` is the
        number of parameters.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        return -2 * self._log_lik + 2 * self._n_params

    def bic(self):
        """Bayesian Information Criterion.

        .. math::

            \\mathrm{BIC} = -2\\ell + p \\ln(n)

        where :math:`n` is the number of observations.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        return -2 * self._log_lik + self._n_params * np.log(self.n)

    def summary(self):
        """Print a formatted summary table.  Must be overridden."""
        raise NotImplementedError


# ======================================================================
# SAR: y = rho*W*y + X*beta + epsilon
# ======================================================================

class SAR(_SpatialModel):
    """Spatial Autoregressive (SAR) model.

    .. math::

        y = \\rho W y + X\\beta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Also known as the **spatial lag model**.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Design matrix.
    W : Weights or ndarray of shape (n, n)
        Row-standardized spatial weights matrix.

    Attributes
    ----------
    beta : ndarray of shape (k,)
        Estimated coefficients.
    rho : float
        Estimated spatial autoregressive parameter.
    sigma2 : float
        Estimated error variance.
    se_beta : ndarray of shape (k,)
        Standard errors of beta.
    se_rho : float
        Standard error of rho.
    """

    def fit(self, rho_bounds=(-0.99, 0.99)):
        """Fit the SAR model via concentrated MLE.

        Parameters
        ----------
        rho_bounds : tuple of float, optional
            Bounds for :math:`\\rho`.

        Returns
        -------
        SAR
            Returns self (for method chaining).
        """
        result = concentrated_mle_sar(self.y, self.X, self.W_obj,
                                       rho_bounds=rho_bounds)

        self.beta = result["beta"]
        self.rho = result["rho"]
        self.sigma2 = result["sigma2"]
        self.se_beta = result["se_beta"]
        self.se_rho = result["se_rho"]
        self._log_lik = result["log_likelihood"]
        self._fitted = True
        self._n_params = self.k + 2  # beta + rho + sigma2
        self.eigvals = result["eigvals"]
        self.convergence = result["convergence"]

        return self

    def predict(self, X_new=None):
        """Generate predicted values.

        For the SAR model, the reduced-form prediction is:

        .. math::

            \\hat{y} = (I - \\hat{\\rho}W)^{-1} X \\hat{\\beta}

        Parameters
        ----------
        X_new : ndarray, optional
            New design matrix.

        Returns
        -------
        ndarray
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if X_new is None:
            X_new = self.X
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        n_new = X_new.shape[0]
        A_inv = np.linalg.inv(np.eye(self.n) - self.rho * self.W)

        # If predicting on new data, need to handle spatial filter
        if X_new.shape[0] == self.n and np.allclose(X_new, self.X):
            return A_inv @ X_new @ self.beta
        else:
            # Out-of-sample: use only the exogenous component
            return X_new @ self.beta

    def summary(self):
        """Print a formatted summary of the SAR model results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")

        lines = []
        lines.append("=" * 68)
        lines.append("Spatial Autoregressive Model (SAR) - MLE Estimation")
        lines.append("=" * 68)
        lines.append(f"Observations:      {self.n}")
        lines.append(f"Variables:         {self.k}")
        lines.append(f"Log-Likelihood:    {self._log_lik:.4f}")
        lines.append(f"AIC:               {self.aic():.4f}")
        lines.append(f"BIC:               {self.bic():.4f}")
        lines.append(f"Sigma^2:           {self.sigma2:.6f}")
        lines.append(f"Converged:         {self.convergence}")
        lines.append("-" * 68)
        lines.append(f"{'Variable':<20s} {'Coef.':>10s} {'Std.Err.':>10s} "
                      f"{'z':>8s} {'P>|z|':>8s}")
        lines.append("-" * 68)

        # Beta estimates
        for i in range(self.k):
            coef = self.beta[i]
            se = self.se_beta[i]
            z = coef / se if se > 0 else 0.0
            p = 2.0 * norm.sf(abs(z))
            stars = _significance_stars(p)
            # Fix: handle variable-length parameter names
            var_name = f"X{i}"
            lines.append(f"{var_name:<20s} {coef:>10.4f} {se:>10.4f} "
                          f"{z:>8.2f} {_format_pvalue(p):>8s} {stars}")

        # Spatial parameter
        z_rho = self.rho / self.se_rho if self.se_rho > 0 else 0.0
        p_rho = 2.0 * norm.sf(abs(z_rho))
        stars_rho = _significance_stars(p_rho)
        lines.append(f"{'rho (W*y)':<20s} {self.rho:>10.4f} {self.se_rho:>10.4f} "
                      f"{z_rho:>8.2f} {_format_pvalue(p_rho):>8s} {stars_rho}")

        lines.append("=" * 68)
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        return "\n".join(lines)


# ======================================================================
# SEM: y = X*beta + u, u = lambda*W*u + epsilon
# ======================================================================

class SEM(_SpatialModel):
    """Spatial Error Model (SEM).

    .. math::

        y &= X\\beta + u \\\\
        u &= \\lambda W u + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Design matrix.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Attributes
    ----------
    beta : ndarray of shape (k,)
        Estimated coefficients.
    lam : float
        Estimated spatial error parameter.
    sigma2 : float
        Estimated error variance.
    """

    def fit(self, lambda_bounds=(-0.99, 0.99)):
        """Fit the SEM model via concentrated MLE.

        Parameters
        ----------
        lambda_bounds : tuple of float, optional
            Bounds for :math:`\\lambda`.

        Returns
        -------
        SEM
            Returns self.
        """
        result = concentrated_mle_sem(self.y, self.X, self.W_obj,
                                       lambda_bounds=lambda_bounds)

        self.beta = result["beta"]
        self.lam = result["lam"]
        self.sigma2 = result["sigma2"]
        self.se_beta = result["se_beta"]
        self.se_lam = result["se_lam"]
        self._log_lik = result["log_likelihood"]
        self._fitted = True
        self._n_params = self.k + 2
        self.eigvals = result["eigvals"]
        self.convergence = result["convergence"]

        return self

    def predict(self, X_new=None):
        """Predict.  In SEM, predictions are simply :math:`X\\hat{\\beta}`."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        if X_new is None:
            X_new = self.X
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        return X_new @ self.beta

    def summary(self):
        """Print a formatted summary of the SEM results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")

        lines = []
        lines.append("=" * 68)
        lines.append("Spatial Error Model (SEM) - MLE Estimation")
        lines.append("=" * 68)
        lines.append(f"Observations:      {self.n}")
        lines.append(f"Variables:         {self.k}")
        lines.append(f"Log-Likelihood:    {self._log_lik:.4f}")
        lines.append(f"AIC:               {self.aic():.4f}")
        lines.append(f"BIC:               {self.bic():.4f}")
        lines.append(f"Sigma^2:           {self.sigma2:.6f}")
        lines.append(f"Converged:         {self.convergence}")
        lines.append("-" * 68)
        lines.append(f"{'Variable':<20s} {'Coef.':>10s} {'Std.Err.':>10s} "
                      f"{'z':>8s} {'P>|z|':>8s}")
        lines.append("-" * 68)

        for i in range(self.k):
            coef = self.beta[i]
            se = self.se_beta[i]
            z = coef / se if se > 0 else 0.0
            p = 2.0 * norm.sf(abs(z))
            stars = _significance_stars(p)
            var_name = f"X{i}"
            lines.append(f"{var_name:<20s} {coef:>10.4f} {se:>10.4f} "
                          f"{z:>8.2f} {_format_pvalue(p):>8s} {stars}")

        z_lam = self.lam / self.se_lam if self.se_lam > 0 else 0.0
        p_lam = 2.0 * norm.sf(abs(z_lam))
        stars_lam = _significance_stars(p_lam)
        lines.append(f"{'lambda':<20s} {self.lam:>10.4f} {self.se_lam:>10.4f} "
                      f"{z_lam:>8.2f} {_format_pvalue(p_lam):>8s} {stars_lam}")

        lines.append("=" * 68)
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        return "\n".join(lines)


# ======================================================================
# SDM: y = rho*W*y + X*beta + W*X*theta + epsilon
# ======================================================================

class SDM(_SpatialModel):
    """Spatial Durbin Model (SDM).

    .. math::

        y = \\rho W y + X\\beta + W X \\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Design matrix.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix.

    Attributes
    ----------
    beta : ndarray of shape (k,)
        Coefficients on X (direct effects component).
    theta : ndarray of shape (k,)
        Coefficients on WX (spatial spillover component).
    rho : float
        Spatial autoregressive parameter.
    sigma2 : float
        Estimated error variance.
    """

    def fit(self, rho_bounds=(-0.99, 0.99)):
        """Fit the SDM model via concentrated MLE.

        Parameters
        ----------
        rho_bounds : tuple of float, optional
            Bounds for :math:`\\rho`.

        Returns
        -------
        SDM
            Returns self.
        """
        result = concentrated_mle_sdm(self.y, self.X, self.W_obj,
                                       rho_bounds=rho_bounds)

        self.beta = result["beta"]
        self.theta = result["theta"]
        self.rho = result["rho"]
        self.sigma2 = result["sigma2"]
        self.se_beta = result["se_beta"]
        self.se_theta = result["se_theta"]
        self.se_rho = result["se_rho"]
        self._log_lik = result["log_likelihood"]
        self._fitted = True
        self._n_params = 2 * self.k + 2  # beta + theta + rho + sigma2
        self.eigvals = result["eigvals"]
        self.convergence = result["convergence"]

        return self

    def predict(self, X_new=None):
        """Generate predictions.  In SDM this involves spatial feedback."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        if X_new is None:
            X_new = self.X
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        n_new = X_new.shape[0]
        if n_new == self.n:
            Z = np.column_stack([X_new, self.W @ X_new])
            gamma = np.concatenate([self.beta, self.theta])
            A_inv = np.linalg.inv(np.eye(self.n) - self.rho * self.W)
            return A_inv @ Z @ gamma
        else:
            return X_new @ self.beta

    def summary(self):
        """Print a formatted summary of the SDM results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")

        lines = []
        lines.append("=" * 68)
        lines.append("Spatial Durbin Model (SDM) - MLE Estimation")
        lines.append("=" * 68)
        lines.append(f"Observations:      {self.n}")
        lines.append(f"Variables:         {self.k}")
        lines.append(f"Log-Likelihood:    {self._log_lik:.4f}")
        lines.append(f"AIC:               {self.aic():.4f}")
        lines.append(f"BIC:               {self.bic():.4f}")
        lines.append(f"Sigma^2:           {self.sigma2:.6f}")
        lines.append(f"Converged:         {self.convergence}")
        lines.append("-" * 68)
        lines.append(f"{'Variable':<20s} {'Coef.':>10s} {'Std.Err.':>10s} "
                      f"{'z':>8s} {'P>|z|':>8s}")
        lines.append("-" * 68)

        # Beta (direct)
        for i in range(self.k):
            coef = self.beta[i]
            se = self.se_beta[i]
            z = coef / se if se > 0 else 0.0
            p = 2.0 * norm.sf(abs(z))
            stars = _significance_stars(p)
            var_name = f"X{i}"
            lines.append(f"{var_name:<20s} {coef:>10.4f} {se:>10.4f} "
                          f"{z:>8.2f} {_format_pvalue(p):>8s} {stars}")

        # Theta (spillover)
        for i in range(self.k):
            coef = self.theta[i]
            se = self.se_theta[i]
            z = coef / se if se > 0 else 0.0
            p = 2.0 * norm.sf(abs(z))
            stars = _significance_stars(p)
            var_name = f"W*X{i}"
            lines.append(f"{var_name:<20s} {coef:>10.4f} {se:>10.4f} "
                          f"{z:>8.2f} {_format_pvalue(p):>8s} {stars}")

        # Rho
        z_rho = self.rho / self.se_rho if self.se_rho > 0 else 0.0
        p_rho = 2.0 * norm.sf(abs(z_rho))
        stars_rho = _significance_stars(p_rho)
        lines.append(f"{'rho (W*y)':<20s} {self.rho:>10.4f} {self.se_rho:>10.4f} "
                      f"{z_rho:>8.2f} {_format_pvalue(p_rho):>8s} {stars_rho}")

        lines.append("=" * 68)
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        return "\n".join(lines)


# ======================================================================
# SAC: y = rho*W*y + X*beta + u, u = lambda*W*u + epsilon
# ======================================================================

class SAC(_SpatialModel):
    """Spatial Autoregressive Combined (SAC) model.

    .. math::

        y &= \\rho W y + X\\beta + u \\\\
        u &= \\lambda W u + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    This model includes both a spatial lag of the dependent variable and
    spatial autocorrelation in the error term (Kelejian--Prucha model).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k)
        Design matrix.
    W : Weights or ndarray of shape (n, n)
        Spatial weights matrix (used for both processes).

    Attributes
    ----------
    beta : ndarray of shape (k,)
        Estimated coefficients.
    rho : float
        Spatial autoregressive parameter.
    lam : float
        Spatial error parameter.
    sigma2 : float
        Estimated error variance.
    """

    def fit(self, rho_bounds=(-0.99, 0.99), lambda_bounds=(-0.99, 0.99)):
        """Fit the SAC model via concentrated MLE.

        Parameters
        ----------
        rho_bounds : tuple of float, optional
            Bounds for :math:`\\rho`.
        lambda_bounds : tuple of float, optional
            Bounds for :math:`\\lambda`.

        Returns
        -------
        SAC
            Returns self.
        """
        result = concentrated_mle_sac(self.y, self.X, self.W_obj,
                                       rho_bounds=rho_bounds,
                                       lambda_bounds=lambda_bounds)

        self.beta = result["beta"]
        self.rho = result["rho"]
        self.lam = result["lam"]
        self.sigma2 = result["sigma2"]
        self.se_beta = result["se_beta"]
        self.se_rho = result["se_rho"]
        self.se_lam = result["se_lam"]
        self._log_lik = result["log_likelihood"]
        self._fitted = True
        self._n_params = self.k + 3  # beta + rho + lambda + sigma2
        self.eigvals = result["eigvals"]
        self.convergence = result["convergence"]

        return self

    def predict(self, X_new=None):
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")
        if X_new is None:
            X_new = self.X
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        n_new = X_new.shape[0]
        if n_new == self.n:
            A = np.linalg.inv(np.eye(self.n) - self.rho * self.W)
            return A @ X_new @ self.beta
        else:
            return X_new @ self.beta

    def summary(self):
        """Print a formatted summary of the SAC results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.")

        lines = []
        lines.append("=" * 68)
        lines.append("Spatial Autoregressive Combined (SAC) - MLE Estimation")
        lines.append("=" * 68)
        lines.append(f"Observations:      {self.n}")
        lines.append(f"Variables:         {self.k}")
        lines.append(f"Log-Likelihood:    {self._log_lik:.4f}")
        lines.append(f"AIC:               {self.aic():.4f}")
        lines.append(f"BIC:               {self.bic():.4f}")
        lines.append(f"Sigma^2:           {self.sigma2:.6f}")
        lines.append(f"Converged:         {self.convergence}")
        lines.append("-" * 68)
        lines.append(f"{'Variable':<20s} {'Coef.':>10s} {'Std.Err.':>10s} "
                      f"{'z':>8s} {'P>|z|':>8s}")
        lines.append("-" * 68)

        for i in range(self.k):
            coef = self.beta[i]
            se = self.se_beta[i]
            z = coef / se if se > 0 else 0.0
            p = 2.0 * norm.sf(abs(z))
            stars = _significance_stars(p)
            var_name = f"X{i}"
            lines.append(f"{var_name:<20s} {coef:>10.4f} {se:>10.4f} "
                          f"{z:>8.2f} {_format_pvalue(p):>8s} {stars}")

        z_rho = self.rho / self.se_rho if self.se_rho > 0 else 0.0
        p_rho = 2.0 * norm.sf(abs(z_rho))
        z_lam = self.lam / self.se_lam if self.se_lam > 0 else 0.0
        p_lam = 2.0 * norm.sf(abs(z_lam))

        stars_rho = _significance_stars(p_rho)
        stars_lam = _significance_stars(p_lam)

        lines.append(f"{'rho (W*y)':<20s} {self.rho:>10.4f} {self.se_rho:>10.4f} "
                      f"{z_rho:>8.2f} {_format_pvalue(p_rho):>8s} {stars_rho}")
        lines.append(f"{'lambda':<20s} {self.lam:>10.4f} {self.se_lam:>10.4f} "
                      f"{z_lam:>8.2f} {_format_pvalue(p_lam):>8s} {stars_lam}")

        lines.append("=" * 68)
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        return "\n".join(lines)

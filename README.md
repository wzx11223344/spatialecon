# SpatialEcon: Spatial Econometrics Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SpatialEcon** is a lightweight, pure-Python toolkit for **spatial econometrics** -- the statistical analysis of data with spatial dependence. It provides tools for spatial autocorrelation diagnostics, spatial regression models estimated via maximum likelihood, and spatial effects decomposition.

> *"Everything is related to everything else, but near things are more related than distant things."* -- Waldo Tobler (1970)

---

## Features

- **Spatial Weights Matrices** -- KNN, distance-based, and contiguity (rook/queen) constructions with row-standardization
- **Spatial Autocorrelation Diagnostics** -- Global Moran's *I*, Geary's *C*, Local Moran's *I* (LISA), and Lagrange Multiplier (LM) tests
- **Spatial Regression Models** -- SAR, SEM, SDM, and SAC estimated via concentrated maximum likelihood
- **Spatial Effects** -- Direct, indirect (spillover), and total effects decomposition (LeSage & Pace 2009)
- **Clean API** -- `fit()` / `summary()` / `predict()` interface with LaTeX formulas in docstrings
- **Lightweight** -- depends only on NumPy, SciPy, and scikit-learn

---

## Installation

```bash
pip install spatialecon
```

Or install from source:

```bash
git clone https://github.com/yourusername/spatialecon.git
cd spatialecon
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from spatialecon import generate_spatial_data, morans_i, lm_lag, SAR

# 1. Generate spatial data on a 20x20 grid
data = generate_spatial_data(
    grid_shape=(20, 20),
    rho=0.65,
    beta=[1.0, 0.5, -0.3, 0.8],
    seed=42,
)

y, X, W = data["y"], data["X"], data["W"]

# 2. Test for spatial autocorrelation
mi = morans_i(y, W)
print(f"Moran's I = {mi.I:.3f}, p = {mi.p_value:.4f}")

# 3. LM specification tests
lm_lag_test = lm_lag(y, X, W)
print(f"LM-lag = {lm_lag_test.statistic:.3f}, p = {lm_lag_test.p_value:.4f}")

# 4. Fit SAR model
model = SAR(y, X, W)
model.fit()
print(model.summary())

# 5. Direct / indirect effects
from spatialecon import direct_indirect_effects
effects = direct_indirect_effects(model)
print(f"Direct effects:   {effects.direct}")
print(f"Indirect effects:  {effects.indirect}")
print(f"Total effects:     {effects.total}")
```

---

## Models

### Spatial Autoregressive (SAR) / Spatial Lag Model

$$y = \rho W y + X\beta + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I)$$

The dependent variable $y$ in each unit depends on the values of $y$ in neighboring units through the spatial lag term $\rho W y$.

### Spatial Error Model (SEM)

$$y = X\beta + u, \quad u = \lambda W u + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I)$$

Spatial dependence is confined to the error term $u$. Neighboring errors are correlated.

### Spatial Durbin Model (SDM)

$$y = \rho W y + X\beta + WX\theta + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I)$$

Extends SAR by including spatially lagged regressors $WX$. This captures both endogenous and exogenous interaction effects.

### Spatial Autoregressive Combined (SAC)

$$y = \rho W_1 y + X\beta + u, \quad u = \lambda W_2 u + \varepsilon$$

AKA the Kelejian-Prucha model. Combines a spatial lag of $y$ with spatial error autocorrelation.

---

## Diagnostics

| Test | Description | Reference |
|------|-------------|-----------|
| `morans_i()` | Global Moran's *I* | Moran (1950) |
| `gearys_c()` | Geary's *C* | Geary (1954) |
| `local_morans_i()` | LISA (Local Moran) | Anselin (1995) |
| `lm_lag()` | LM test for spatial lag | Anselin (1988) |
| `lm_error()` | LM test for spatial error | Anselin (1988) |
| `robust_lm_lag()` | Robust LM-lag | Anselin et al. (1996) |
| `robust_lm_error()` | Robust LM-error | Anselin et al. (1996) |

### LM Test Decision Rule

Following Anselin & Florax (1995):

1. Estimate OLS: $y = X\beta + \varepsilon$
2. Compute LM-lag and LM-error
3. If **only** LM-lag is significant $\rightarrow$ SAR
4. If **only** LM-error is significant $\rightarrow$ SEM
5. If **both** significant $\rightarrow$ compare **robust** LM tests; choose the one with larger statistic (or more significant)

---

## Estimation Methodology

All models are estimated via **concentrated maximum likelihood** (MLE). The log-likelihood function is profiled with respect to $\beta$ and $\sigma^2$, reducing the optimization to a one-dimensional search over the spatial parameter ($\rho$ or $\lambda$):

$$\ell_c(\rho) = -\frac{n}{2}\left[\ln(2\pi) + 1\right] - \frac{n}{2}\ln\hat{\sigma}^2(\rho) + \ln|I - \rho W|$$

The Jacobian term $\ln|I - \rho W|$ is efficiently computed from the eigenvalues of $W$:

$$\ln|I - \rho W| = \sum_{i=1}^n \ln(1 - \rho\lambda_i)$$

---

## Package Structure

```
spatialecon/
├── README.md
├── setup.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── spatialecon/
│   ├── __init__.py         # Package exports
│   ├── weights.py          # Spatial weights matrices
│   ├── diagnostics.py      # Moran's I, Geary's C, LISA, LM tests
│   ├── models.py           # SAR, SEM, SDM, SAC model classes
│   ├── estimation.py       # Concentrated MLE routines
│   └── utils.py            # Data generation, effects decomposition
└── examples/
    └── demo.py             # Full demonstration
```

---

## Dependencies

- `numpy >= 1.20.0`
- `scipy >= 1.7.0`
- `scikit-learn >= 0.24.0`

---

## Roadmap

- [ ] Spatial panel data models
- [ ] GMM estimation (Kelejian-Prucha)
- [ ] Bayesian spatial models (MCMC)
- [ ] Spatial probit / logit
- [ ] Spatio-temporal models
- [ ] Prediction intervals and cross-validation
- [ ] Support for sparse weights matrices
- [ ] Full standard errors via analytical information matrix

---

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L. (1995). Local Indicators of Spatial Association -- LISA. *Geographical Analysis*, 27(2), 93-115.
- Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics*, 26(1), 77-104.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes: Models & Applications*. Pion.
- LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17-23.

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on GitHub.

---

**Made with NumPy, SciPy, and spatial thinking.**
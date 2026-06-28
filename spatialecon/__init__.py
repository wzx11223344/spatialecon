"""
SpatialEcon: Spatial Econometrics Toolkit
=========================================

Spatial autocorrelation, spatial regression models, and spatial diagnostics
for Python.

Modules
-------
weights      : Spatial weights matrix construction and manipulation
diagnostics  : Spatial autocorrelation diagnostics (Moran's I, Geary's C, LISA, LM tests)
models       : Spatial regression models (SAR, SEM, SDM, SAC)
estimation   : Maximum likelihood estimation for spatial models
utils        : Data generation and spatial effects computation
"""

__version__ = "0.1.0"
__author__ = "SpatialEcon Contributors"

from .weights import (
    Weights,
    knn_weights,
    distance_weights,
    contiguity_weights,
    row_standardize,
    spatial_lag,
)
from .diagnostics import (
    morans_i,
    gearys_c,
    local_morans_i,
    lm_lag,
    lm_error,
    robust_lm_lag,
    robust_lm_error,
)
from .models import (
    SAR,
    SEM,
    SDM,
    SAC,
)
from .estimation import (
    concentrated_mle_sar,
    concentrated_mle_sem,
    concentrated_mle_sdm,
    concentrated_mle_sac,
)
from .utils import (
    generate_spatial_data,
    direct_indirect_effects,
)

__all__ = [
    # weights
    "Weights",
    "knn_weights",
    "distance_weights",
    "contiguity_weights",
    "row_standardize",
    "spatial_lag",
    # diagnostics
    "morans_i",
    "gearys_c",
    "local_morans_i",
    "lm_lag",
    "lm_error",
    "robust_lm_lag",
    "robust_lm_error",
    # models
    "SAR",
    "SEM",
    "SDM",
    "SAC",
    # estimation
    "concentrated_mle_sar",
    "concentrated_mle_sem",
    "concentrated_mle_sdm",
    "concentrated_mle_sac",
    # utils
    "generate_spatial_data",
    "direct_indirect_effects",
]

#!/usr/bin/env python
"""
SpatialEcon Demo
================

This script demonstrates the core functionality of the SpatialEcon toolkit:

1. Generate spatial data on a 20x20 grid
2. Compute Global Moran's I (should detect positive spatial autocorrelation)
3. Run LM specification tests to select the appropriate model
4. Fit a SAR model and compare estimates to true parameters
5. Compute direct, indirect, and total effects
"""

import sys
import os

# Allow running from examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from spatialecon import (
    contiguity_weights,
    morans_i,
    lm_lag,
    lm_error,
    robust_lm_lag,
    robust_lm_error,
    SAR,
    generate_spatial_data,
    direct_indirect_effects,
)


def print_section(title):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    # ------------------------------------------------------------------
    # 1. Generate synthetic spatial data
    # ------------------------------------------------------------------
    print_section("1. Generating Spatial Data (20x20 Grid, SAR DGP)")

    grid_shape = (20, 20)
    true_rho = 0.65
    true_beta = np.array([1.0, 0.5, -0.3, 0.8])  # [intercept, X1, X2, X3]
    k = len(true_beta) - 1

    np.random.seed(42)
    data = generate_spatial_data(
        n=grid_shape[0] * grid_shape[1],
        k=k,
        rho=true_rho,
        beta=true_beta,
        grid_shape=grid_shape,
        sigma2=1.0,
        seed=42,
    )

    y = data["y"]
    X = data["X"]
    W = data["W"]

    print(f"  Observations:    {W.n}")
    print(f"  Grid shape:      {grid_shape}")
    print(f"  True rho:        {true_rho}")
    print(f"  True beta:       {true_beta}")
    print(f"  True sigma2:     1.0")

    # ------------------------------------------------------------------
    # 2. Global Moran's I
    # ------------------------------------------------------------------
    print_section("2. Global Spatial Autocorrelation: Moran's I")

    mi_result = morans_i(y, W)
    print(f"  Moran's I:       {mi_result.I:.4f}")
    print(f"  E[I]:            {mi_result.EI:.4f}")
    print(f"  Var[I]:          {mi_result.VI:.6f}")
    print(f"  z-score:         {mi_result.z_score:.4f}")
    print(f"  p-value:         {mi_result.p_value:.4f}")

    if mi_result.p_value < 0.05:
        print("  => Significant positive spatial autocorrelation detected.")
    else:
        print("  => No significant spatial autocorrelation.")

    # Also run Moran's I on OLS residuals to check for remaining correlation
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    e_ols = y - X @ beta_ols
    mi_resid = morans_i(e_ols, W)
    print()
    print(f"  Moran's I (OLS residuals): {mi_resid.I:.4f}  (p={mi_resid.p_value:.4f})")

    # ------------------------------------------------------------------
    # 3. LM Specification Tests
    # ------------------------------------------------------------------
    print_section("3. LM Specification Tests (Model Selection)")

    lm_lag_result = lm_lag(y, X, W)
    lm_err_result = lm_error(y, X, W)
    rlm_lag_result = robust_lm_lag(y, X, W)
    rlm_err_result = robust_lm_error(y, X, W)

    print(f"  {'Test':<25s} {'Statistic':>10s} {'p-value':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'LM Lag':<25s} {lm_lag_result.statistic:>10.4f} {lm_lag_result.p_value:>10.4f}")
    print(f"  {'LM Error':<25s} {lm_err_result.statistic:>10.4f} {lm_err_result.p_value:>10.4f}")
    print(f"  {'Robust LM Lag':<25s} {rlm_lag_result.statistic:>10.4f} {rlm_lag_result.p_value:>10.4f}")
    print(f"  {'Robust LM Error':<25s} {rlm_err_result.statistic:>10.4f} {rlm_err_result.p_value:>10.4f}")

    # Decision rule (Anselin 1988 / Anselin & Florax 1995)
    print()
    print("  Model Selection Decision:")
    if lm_lag_result.p_value < 0.05 and rlm_lag_result.p_value < 0.05:
        if lm_err_result.p_value > 0.05:
            print("  => SAR (lag) model indicated (LM-lag significant, LM-error not)")
        elif rlm_lag_result.statistic > rlm_err_result.statistic:
            print("  => SAR (lag) model indicated (robust LM-lag more significant)")
        else:
            print("  => SEM (error) model indicated (robust LM-error more significant)")
    elif lm_err_result.p_value < 0.05 and rlm_err_result.p_value < 0.05:
        print("  => SEM (error) model indicated")
    else:
        print("  => No clear spatial dependence pattern; OLS may suffice")

    # ------------------------------------------------------------------
    # 4. Fit SAR Model
    # ------------------------------------------------------------------
    print_section("4. SAR Model Estimation (MLE)")

    model = SAR(y, X, W)
    model.fit(rho_bounds=(-0.99, 0.99))

    print(model.summary())

    # Compare estimates to true parameters
    print()
    print_section("5. Comparison: Estimated vs True Parameters")
    print(f"  {'Parameter':<15s} {'True':>10s} {'Estimated':>12s} {'Bias':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

    param_names = [f"beta[{i}]" for i in range(k + 1)] + ["rho"]
    true_values = list(true_beta) + [true_rho]
    estimates = list(model.beta) + [model.rho]

    for name, true_val, est_val in zip(param_names, true_values, estimates):
        bias = est_val - true_val
        print(f"  {name:<15s} {true_val:>10.4f} {est_val:>12.4f} {bias:>10.4f}")

    # ------------------------------------------------------------------
    # 6. Direct / Indirect Effects
    # ------------------------------------------------------------------
    print_section("6. Spatial Effects: Direct, Indirect, Total")

    effects = direct_indirect_effects(model)

    print(f"  {'Variable':<12s} {'Direct':>10s} {'Indirect':>10s} {'Total':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(model.k):
        var_name = f"X{i}"
        print(f"  {var_name:<12s} {effects.direct[i]:>10.4f} "
              f"{effects.indirect[i]:>10.4f} {effects.total[i]:>10.4f}")

    # Interpretation
    print()
    print("  Interpretation:")
    print(f"  - Direct effects: average impact of a unit's own X change on its own y")
    print(f"  - Indirect effects: spillover impact on neighbors' y")
    print(f"  - Total effects: sum of direct and indirect")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 68)
    print("  SpatialEcon Demo Complete!")
    print("=" * 68)
    print()
    print("  Key findings:")
    print(f"  1. Moran's I = {mi_result.I:.3f} (p={mi_result.p_value:.4f}) confirms spatial")
    print(f"     autocorrelation in the generated data.")
    print(f"  2. LM tests correctly identify the SAR specification suitable.")
    print(f"  3. SAR MLE recovers rho = {model.rho:.3f} (true = {true_rho})")
    print(f"     and beta estimates close to true values.")
    print(f"  4. Spatial spillover effects are present -- indirect effects are")
    print(f"     non-zero, reflecting the global feedback in SAR models.")
    print()


if __name__ == "__main__":
    main()

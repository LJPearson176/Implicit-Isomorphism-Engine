from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - optional dependency in CI runtime
    cp = None

from portfolio_proof_lab.types import OptimizationConfig, OptimizationInputs


def _validate_inputs(inputs: OptimizationInputs) -> None:
    n = inputs.expected_returns.shape[0]
    if n == 0:
        raise ValueError("expected_returns cannot be empty")
    if inputs.covariance.shape != (n, n):
        raise ValueError("covariance must be square with matching asset dimension")
    if inputs.current_weights.shape != (n,):
        raise ValueError("current_weights dimension mismatch")
    if inputs.cost_rates.shape != (n,):
        raise ValueError("cost_rates dimension mismatch")

    for name, arr in {
        "expected_returns": inputs.expected_returns,
        "covariance": inputs.covariance,
        "current_weights": inputs.current_weights,
        "cost_rates": inputs.cost_rates,
    }.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")

    if not np.allclose(inputs.covariance, inputs.covariance.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    min_eig = float(np.min(np.linalg.eigvalsh(inputs.covariance)))
    if min_eig < -1e-8:
        raise ValueError("covariance must be positive semidefinite within tolerance")

    cfg = inputs.config
    if cfg.max_weight <= 0 or cfg.max_weight > 1:
        raise ValueError("max_weight must be in (0, 1]")
    if cfg.max_turnover < 0:
        raise ValueError("max_turnover must be non-negative")
    if cfg.max_variance < 0:
        raise ValueError("max_variance must be non-negative")
    if not math.isfinite(cfg.risk_aversion) or cfg.risk_aversion < 0:
        raise ValueError("risk_aversion must be finite and non-negative")
    if not math.isfinite(cfg.turnover_penalty) or cfg.turnover_penalty < 0:
        raise ValueError("turnover_penalty must be finite and non-negative")


def build_problem(inputs: OptimizationInputs) -> Tuple["cp.Problem", "cp.Variable"]:
    if cp is None:
        raise RuntimeError("cvxpy is not installed")

    _validate_inputs(inputs)
    cfg: OptimizationConfig = inputs.config

    n = inputs.expected_returns.shape[0]
    w = cp.Variable(n)

    expected_term = inputs.expected_returns @ w
    risk_term = 0.5 * cfg.risk_aversion * cp.quad_form(w, inputs.covariance)
    turnover_vector = cp.abs(w - inputs.current_weights)
    cost_term = cfg.turnover_penalty * cp.sum(cp.multiply(inputs.cost_rates, turnover_vector))

    objective = cp.Maximize(expected_term - risk_term - cost_term)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= cfg.max_weight,
        cp.sum(turnover_vector) <= cfg.max_turnover,
        cp.quad_form(w, inputs.covariance) <= cfg.max_variance,
    ]

    return cp.Problem(objective, constraints), w

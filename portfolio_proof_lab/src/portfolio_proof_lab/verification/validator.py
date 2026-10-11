from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from portfolio_proof_lab.types import OptimizationInputs, OptimizationResult, ValidationResult


def _independent_objective(weights: np.ndarray, inputs: OptimizationInputs) -> float:
    cfg = inputs.config
    expected = float(inputs.expected_returns @ weights)
    risk = 0.5 * cfg.risk_aversion * float(weights.T @ inputs.covariance @ weights)
    turnover = np.abs(weights - inputs.current_weights)
    cost = cfg.turnover_penalty * float(np.sum(inputs.cost_rates * turnover))
    return expected - risk - cost


def validate_allocation(result: OptimizationResult, inputs: OptimizationInputs) -> ValidationResult:
    cfg = inputs.config
    errors: List[str] = []
    checks: Dict[str, float | bool | str] = {}

    w = result.weights
    if w.shape != inputs.current_weights.shape:
        errors.append("weights shape mismatch")
        return ValidationResult(False, errors, checks)

    finite_ok = bool(np.isfinite(w).all())
    checks["finite_weights"] = finite_ok
    if not finite_ok:
        errors.append("weights contain non-finite values")

    sum_error = abs(float(np.sum(w)) - 1.0)
    checks["weight_sum_error"] = sum_error
    if sum_error > cfg.weight_sum_tolerance:
        errors.append("weight sum outside tolerance")

    lower_bound_ok = bool(np.all(w >= -cfg.bound_tolerance))
    upper_bound_ok = bool(np.all(w <= cfg.max_weight + cfg.bound_tolerance))
    checks["lower_bound_ok"] = lower_bound_ok
    checks["upper_bound_ok"] = upper_bound_ok
    if not lower_bound_ok:
        errors.append("lower bound violated")
    if not upper_bound_ok:
        errors.append("upper bound violated")

    turnover = float(np.sum(np.abs(w - inputs.current_weights)))
    checks["turnover"] = turnover
    if turnover > cfg.max_turnover + cfg.bound_tolerance:
        errors.append("turnover limit violated")

    variance = float(w.T @ inputs.covariance @ w)
    checks["variance"] = variance
    if variance > cfg.max_variance + cfg.bound_tolerance:
        errors.append("variance limit violated")

    objective_independent = _independent_objective(w, inputs)
    checks["objective_independent"] = objective_independent
    checks["objective_reported"] = result.objective_value

    if math.isfinite(result.objective_value):
        objective_gap = abs(objective_independent - result.objective_value)
        checks["objective_gap"] = objective_gap
        if objective_gap > 1e-4:
            errors.append("objective mismatch")
    else:
        errors.append("reported objective is not finite")

    status_ok = result.status in cfg.accepted_statuses
    checks["status"] = result.status
    checks["status_ok"] = status_ok
    if not status_ok:
        errors.append("solver status unacceptable")

    checks["covariance_psd"] = bool(np.min(np.linalg.eigvalsh(inputs.covariance)) >= -1e-8)
    if not checks["covariance_psd"]:
        errors.append("covariance fails PSD tolerance")

    return ValidationResult(is_valid=not errors, errors=errors, checks=checks)

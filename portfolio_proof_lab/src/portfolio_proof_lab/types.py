from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class OptimizationConfig:
    risk_aversion: float
    turnover_penalty: float
    max_weight: float
    max_turnover: float
    max_variance: float
    weight_sum_tolerance: float = 1e-6
    bound_tolerance: float = 1e-8
    solver: str = "SCS"
    solver_max_iters: int = 10_000
    solver_eps: float = 1e-6
    accepted_statuses: tuple[str, ...] = ("optimal", "optimal_inaccurate")


@dataclass(frozen=True)
class OptimizationInputs:
    expected_returns: np.ndarray
    covariance: np.ndarray
    current_weights: np.ndarray
    cost_rates: np.ndarray
    config: OptimizationConfig


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    status: str
    objective_value: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: List[str]
    checks: Dict[str, Any]

from __future__ import annotations

import importlib.metadata

import numpy as np

from portfolio_proof_lab.optimizer.model import build_problem
from portfolio_proof_lab.types import OptimizationInputs, OptimizationResult


def optimize_portfolio(inputs: OptimizationInputs) -> OptimizationResult:
    problem, w_var = build_problem(inputs)
    cfg = inputs.config

    objective_value = problem.solve(
        solver=cfg.solver,
        max_iters=cfg.solver_max_iters,
        eps=cfg.solver_eps,
        verbose=False,
    )

    weights = np.asarray(w_var.value, dtype=float) if w_var.value is not None else np.array([])

    diagnostics = {
        "solver": cfg.solver,
        "solver_max_iters": cfg.solver_max_iters,
        "solver_eps": cfg.solver_eps,
        "solver_stats": {
            "num_iters": getattr(problem.solver_stats, "num_iters", None),
            "solve_time": getattr(problem.solver_stats, "solve_time", None),
        },
        "primal_residual": getattr(problem.solver_stats, "extra_stats", {}),
        "cvxpy_version": importlib.metadata.version("cvxpy") if weights.size else None,
    }

    return OptimizationResult(
        weights=weights,
        status=str(problem.status).lower(),
        objective_value=float(objective_value) if objective_value is not None else float("nan"),
        diagnostics=diagnostics,
    )

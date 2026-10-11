import unittest

import numpy as np

from portfolio_proof_lab.types import OptimizationConfig, OptimizationInputs, OptimizationResult
from portfolio_proof_lab.verification.validator import validate_allocation


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.config = OptimizationConfig(
            risk_aversion=1.0,
            turnover_penalty=0.0,
            max_weight=0.7,
            max_turnover=1.0,
            max_variance=1.0,
        )
        self.inputs = OptimizationInputs(
            expected_returns=np.array([0.02, 0.01]),
            covariance=np.array([[0.1, 0.0], [0.0, 0.1]]),
            current_weights=np.array([0.5, 0.5]),
            cost_rates=np.array([0.0, 0.0]),
            config=self.config,
        )

    def test_accepts_valid_allocation(self):
        weights = np.array([0.6, 0.4])
        objective = float(
            self.inputs.expected_returns @ weights
            - 0.5 * self.config.risk_aversion * (weights.T @ self.inputs.covariance @ weights)
        )
        result = OptimizationResult(
            weights=weights,
            status="optimal",
            objective_value=objective,
            diagnostics={},
        )
        validation = validate_allocation(result, self.inputs)
        self.assertTrue(validation.is_valid)

    def test_rejects_bound_violation(self):
        result = OptimizationResult(
            weights=np.array([0.9, 0.1]),
            status="optimal",
            objective_value=0.0,
            diagnostics={},
        )
        validation = validate_allocation(result, self.inputs)
        self.assertFalse(validation.is_valid)
        self.assertIn("upper bound violated", validation.errors)


if __name__ == "__main__":
    unittest.main()

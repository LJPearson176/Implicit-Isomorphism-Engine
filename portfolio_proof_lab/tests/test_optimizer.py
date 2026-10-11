import unittest

import numpy as np

from portfolio_proof_lab.optimizer.model import cp
from portfolio_proof_lab.optimizer.service import optimize_portfolio
from portfolio_proof_lab.types import OptimizationConfig, OptimizationInputs


@unittest.skipIf(cp is None, "cvxpy not available")
class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = OptimizationConfig(
            risk_aversion=1.0,
            turnover_penalty=0.5,
            max_weight=0.8,
            max_turnover=1.0,
            max_variance=1.0,
        )

    def test_optimizer_returns_feasible_weights(self):
        inputs = OptimizationInputs(
            expected_returns=np.array([0.02, 0.01, 0.005]),
            covariance=np.array(
                [
                    [0.10, 0.01, 0.0],
                    [0.01, 0.08, 0.01],
                    [0.0, 0.01, 0.05],
                ]
            ),
            current_weights=np.array([0.34, 0.33, 0.33]),
            cost_rates=np.array([0.001, 0.001, 0.001]),
            config=self.config,
        )

        result = optimize_portfolio(inputs)
        self.assertIn(result.status, self.config.accepted_statuses)
        self.assertTrue(np.isfinite(result.weights).all())
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=4)

    def test_invalid_covariance_rejected(self):
        inputs = OptimizationInputs(
            expected_returns=np.array([0.02, 0.01]),
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            current_weights=np.array([0.5, 0.5]),
            cost_rates=np.array([0.001, 0.001]),
            config=self.config,
        )

        with self.assertRaises(ValueError):
            optimize_portfolio(inputs)


if __name__ == "__main__":
    unittest.main()

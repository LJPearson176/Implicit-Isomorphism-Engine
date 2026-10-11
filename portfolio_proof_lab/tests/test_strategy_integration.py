import unittest

import numpy as np

from portfolio_proof_lab.strategy.nautilus_adapter import InMemoryPortfolioGateway
from portfolio_proof_lab.strategy.rebalance import ScheduledRebalanceService
from portfolio_proof_lab.types import OptimizationConfig, OptimizationInputs, OptimizationResult
from portfolio_proof_lab.verification.validator import validate_allocation


class TestStrategyIntegration(unittest.TestCase):
    def setUp(self):
        self.symbols = ["SPY", "QQQ"]
        self.config = OptimizationConfig(
            risk_aversion=1.0,
            turnover_penalty=0.0,
            max_weight=0.8,
            max_turnover=1.0,
            max_variance=1.0,
        )

    def _inputs_factory(self, current_weights: np.ndarray) -> OptimizationInputs:
        return OptimizationInputs(
            expected_returns=np.array([0.02, 0.01]),
            covariance=np.array([[0.05, 0.0], [0.0, 0.05]]),
            current_weights=current_weights,
            cost_rates=np.array([0.0, 0.0]),
            config=self.config,
        )

    def test_fail_closed_on_invalid_allocation(self):
        def bad_optimizer(_: OptimizationInputs) -> OptimizationResult:
            return OptimizationResult(
                weights=np.array([1.2, -0.2]),
                status="optimal",
                objective_value=0.0,
                diagnostics={},
            )

        gateway = InMemoryPortfolioGateway(
            _equity=100_000,
            _weights={"SPY": 0.5, "QQQ": 0.5},
            _prices={"SPY": 500, "QQQ": 400},
            submitted_orders=[],
        )

        service = ScheduledRebalanceService(self.symbols, bad_optimizer, validate_allocation)
        decision = service.run(
            gateway,
            self._inputs_factory,
            min_order_qty={"SPY": 1, "QQQ": 1},
            qty_precision={"SPY": 0, "QQQ": 0},
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(len(gateway.submitted_orders), 0)

    def test_submits_orders_when_valid(self):
        def good_optimizer(_: OptimizationInputs) -> OptimizationResult:
            weights = np.array([0.6, 0.4])
            objective = float(
                np.array([0.02, 0.01]) @ weights
                - 0.5 * (weights.T @ np.array([[0.05, 0.0], [0.0, 0.05]]) @ weights)
            )
            return OptimizationResult(
                weights=weights,
                status="optimal",
                objective_value=objective,
                diagnostics={},
            )

        gateway = InMemoryPortfolioGateway(
            _equity=100_000,
            _weights={"SPY": 0.5, "QQQ": 0.5},
            _prices={"SPY": 500, "QQQ": 400},
            submitted_orders=[],
        )

        service = ScheduledRebalanceService(self.symbols, good_optimizer, validate_allocation)
        decision = service.run(
            gateway,
            self._inputs_factory,
            min_order_qty={"SPY": 1, "QQQ": 1},
            qty_precision={"SPY": 0, "QQQ": 0},
        )

        self.assertTrue(decision.accepted)
        self.assertGreater(len(gateway.submitted_orders), 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Protocol

import numpy as np

from portfolio_proof_lab.types import OptimizationInputs, OptimizationResult, ValidationResult


@dataclass(frozen=True)
class OrderInstruction:
    symbol: str
    quantity: float
    target_weight: float


class PortfolioGateway(Protocol):
    def equity(self) -> float: ...
    def current_weights(self, symbols: Iterable[str]) -> np.ndarray: ...
    def prices(self, symbols: Iterable[str]) -> np.ndarray: ...
    def submit_orders(self, orders: List[OrderInstruction]) -> None: ...


@dataclass
class RebalanceDecision:
    accepted: bool
    reason: str
    orders: List[OrderInstruction]
    optimization: OptimizationResult | None
    validation: ValidationResult | None


class ScheduledRebalanceService:
    def __init__(
        self,
        symbols: List[str],
        optimizer: Callable[[OptimizationInputs], OptimizationResult],
        validator: Callable[[OptimizationResult, OptimizationInputs], ValidationResult],
    ):
        self.symbols = symbols
        self.optimizer = optimizer
        self.validator = validator

    def run(
        self,
        gateway: PortfolioGateway,
        optimization_inputs_factory: Callable[[np.ndarray], OptimizationInputs],
        min_order_qty: Dict[str, float],
        qty_precision: Dict[str, int],
    ) -> RebalanceDecision:
        current_weights = gateway.current_weights(self.symbols)
        inputs = optimization_inputs_factory(current_weights)
        result = self.optimizer(inputs)
        validation = self.validator(result, inputs)

        if not validation.is_valid:
            return RebalanceDecision(False, "allocation validation failed", [], result, validation)

        prices = gateway.prices(self.symbols)
        equity = gateway.equity()
        if np.any(prices <= 0) or equity <= 0:
            return RebalanceDecision(False, "invalid prices or equity", [], result, validation)

        current_notional = current_weights * equity
        target_notional = result.weights * equity
        delta_notional = target_notional - current_notional
        raw_qty = delta_notional / prices

        orders: List[OrderInstruction] = []
        for i, symbol in enumerate(self.symbols):
            precision = qty_precision.get(symbol, 0)
            qty = float(np.round(raw_qty[i], precision))
            if abs(qty) < min_order_qty.get(symbol, 0.0):
                continue
            orders.append(OrderInstruction(symbol=symbol, quantity=qty, target_weight=float(result.weights[i])))

        gateway.submit_orders(orders)
        return RebalanceDecision(True, "accepted", orders, result, validation)

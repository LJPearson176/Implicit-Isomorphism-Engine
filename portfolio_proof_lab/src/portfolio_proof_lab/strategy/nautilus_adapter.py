from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from portfolio_proof_lab.strategy.rebalance import OrderInstruction, PortfolioGateway


@dataclass
class InMemoryPortfolioGateway(PortfolioGateway):
    """Backtest-safe boundary mirroring the NautilusTrader integration contract."""

    _equity: float
    _weights: Dict[str, float]
    _prices: Dict[str, float]
    submitted_orders: List[OrderInstruction]

    def equity(self) -> float:
        return float(self._equity)

    def current_weights(self, symbols: Iterable[str]) -> np.ndarray:
        return np.array([self._weights.get(symbol, 0.0) for symbol in symbols], dtype=float)

    def prices(self, symbols: Iterable[str]) -> np.ndarray:
        return np.array([self._prices[symbol] for symbol in symbols], dtype=float)

    def submit_orders(self, orders: List[OrderInstruction]) -> None:
        self.submitted_orders.extend(orders)

    def apply_fill(self, symbol: str, fill_qty: float, fill_price: float) -> None:
        gross = self._weights.get(symbol, 0.0) * self._equity
        gross += fill_qty * fill_price
        self._weights[symbol] = gross / self._equity

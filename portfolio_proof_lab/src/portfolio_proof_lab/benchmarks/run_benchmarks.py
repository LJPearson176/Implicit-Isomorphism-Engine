from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cumulative_return: float
    volatility: float
    sharpe: float
    max_drawdown: float


def _max_drawdown(series: pd.Series) -> float:
    peaks = series.cummax()
    drawdowns = (series - peaks) / peaks
    return float(drawdowns.min())


def evaluate_strategy(returns: pd.DataFrame, weights: np.ndarray, name: str) -> BenchmarkResult:
    portfolio_returns = returns.to_numpy() @ weights
    pnl = pd.Series((1.0 + portfolio_returns).cumprod())
    vol = float(np.std(portfolio_returns) * np.sqrt(252))
    mean = float(np.mean(portfolio_returns) * 252)
    sharpe = mean / vol if vol > 0 else 0.0
    return BenchmarkResult(
        name=name,
        cumulative_return=float(pnl.iloc[-1] - 1.0),
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(pnl),
    )


def run_controlled_benchmarks(returns: pd.DataFrame, fixed_signal_weights: np.ndarray, optimized_weights: np.ndarray) -> Dict[str, BenchmarkResult]:
    n = returns.shape[1]
    equal_weight = np.full(n, 1.0 / n)
    min_var_proxy = np.linalg.pinv(returns.cov().to_numpy())
    one = np.ones(n)
    min_var_weights = (min_var_proxy @ one) / (one @ min_var_proxy @ one)

    return {
        "equal_weight": evaluate_strategy(returns, equal_weight, "equal_weight"),
        "minimum_variance": evaluate_strategy(returns, min_var_weights, "minimum_variance"),
        "optimized_fixed_signal": evaluate_strategy(returns, optimized_weights, "optimized_fixed_signal"),
        "signal_non_optimized_rule": evaluate_strategy(returns, fixed_signal_weights, "signal_non_optimized_rule"),
    }

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_expected_returns(returns: pd.DataFrame, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(returns) < window:
        raise ValueError("insufficient rows for return estimation window")
    return returns.tail(window).mean().to_numpy(dtype=float)


def estimate_covariance(returns: pd.DataFrame, window: int) -> np.ndarray:
    if window <= 1:
        raise ValueError("covariance window must be greater than 1")
    if len(returns) < window:
        raise ValueError("insufficient rows for covariance estimation window")
    cov = returns.tail(window).cov().to_numpy(dtype=float)
    return 0.5 * (cov + cov.T)


def assert_no_lookahead(feature_timestamps: pd.Series, decision_timestamp: pd.Timestamp) -> None:
    if (feature_timestamps > decision_timestamp).any():
        raise ValueError("lookahead detected: feature timestamp exceeds decision timestamp")

"""Vectorized, cost-aware backtester with one-bar execution lag.

Convention
----------
- `signal_t` is the desired position decided at the CLOSE of bar t using
  information available up to t only.
- The position is held from close_t to close_{t+1}, earning `ret_{t+1}`
  (one-bar execution lag -> zero look-ahead).
- Costs: (fee + slippage) bps charged on every unit of |position change|.
- Shorts pay a per-bar funding haircut (perp-style), longs do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(signal: pd.Series, ret: pd.Series, cost_bps: float = 20.0,
             short_funding_bps: float = 0.0) -> pd.DataFrame:
    """Run the backtest. `signal` and `ret` must share an index.

    Returns DataFrame with position, turnover, cost, strategy return, equity.
    """
    sig = signal.reindex(ret.index).astype(float).clip(-1, 1).fillna(0.0)
    pos = sig.shift(1).fillna(0.0)            # decided at t, applied to ret_{t+1}
    turnover = pos.diff().abs().fillna(pos.abs())  # first bar entry counts
    cost = turnover * cost_bps / 1e4
    funding = np.where(pos < 0, short_funding_bps / 1e4 * pos.abs(), 0.0)
    # an unleveraged bar cannot lose more than the capital committed to it;
    # flooring at -1 keeps equity non-negative (cumprod can't flip sign)
    strat_ret = (pos * ret - cost - funding).clip(lower=-1.0)
    equity = (1 + strat_ret).cumprod()
    return pd.DataFrame({
        "position": pos, "turnover": turnover, "cost": cost,
        "strat_ret": strat_ret, "equity": equity, "asset_ret": ret,
    })


def buy_and_hold(ret: pd.Series, cost_bps: float = 20.0) -> pd.DataFrame:
    """Benchmark: fully long the whole window, net of one entry cost."""
    sig = pd.Series(1.0, index=ret.index)
    return backtest(sig, ret, cost_bps=cost_bps, short_funding_bps=0.0)


def proba_to_signal(p_up: pd.Series, mode: str = "long_only", sizing: str = "binary",
                    thr_long: float = 0.55, thr_short: float = 0.45,
                    scale: float = 0.2) -> pd.Series:
    """Map P(up) -> target position.

    binary:  long_only -> {0, 1};  long_short -> {-1, 0, +1}
    scaled:  position proportional to conviction (p-0.5)/scale, clipped.
    kelly:   half-Kelly fraction: f = clip(2p-1, 0, 1) for long_only;
             f = clip(2p-1, -1, 1) for long_short.
    """
    p = p_up.astype(float)
    if sizing == "binary":
        if mode == "long_only":
            return (p > thr_long).astype(float)
        sig = pd.Series(0.0, index=p.index)
        sig[p > thr_long] = 1.0
        sig[p < thr_short] = -1.0
        return sig
    if sizing == "kelly":
        raw = (2 * p - 1)           # full Kelly for win-rate-based edge
        if mode == "long_only":
            return raw.clip(0.0, 1.0)
        return raw.clip(-1.0, 1.0)
    # scaled (default fallback)
    raw = (p - 0.5) / max(scale, 1e-9)
    if mode == "long_only":
        return raw.clip(0.0, 1.0)
    return raw.clip(-1.0, 1.0)

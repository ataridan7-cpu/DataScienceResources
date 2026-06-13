"""Robustness / anti-overfitting diagnostics for the selected strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import perf_metrics


def cost_sweep(signal: pd.Series, ret: pd.Series, ppy: int = 365,
               cost_grid=(0, 5, 10, 20, 40), short_funding_bps: float = 0.0) -> pd.DataFrame:
    """Net performance as a function of per-side cost (bps)."""
    rows = {}
    for c in cost_grid:
        m = perf_metrics(backtest(signal, ret, float(c), short_funding_bps), ppy)
        rows[c] = {"total_return": m["total_return"], "sharpe": m["sharpe"],
                   "max_drawdown": m["max_drawdown"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps_per_side"
    return out


def circular_shift_null(signal: pd.Series, ret: pd.Series, cost_bps: float,
                        n: int = 500, seed: int = 42, ppy: int = 365,
                        short_funding_bps: float = 0.0) -> dict:
    """Null distribution: same signal, circularly shifted by random offsets.

    Preserves the signal's autocorrelation/turnover but destroys any alignment
    with future returns. The real strategy must beat this distribution.
    """
    rng = np.random.default_rng(seed)
    vals = signal.to_numpy()
    real = perf_metrics(backtest(signal, ret, cost_bps, short_funding_bps), ppy)["total_return"]
    nulls = []
    for _ in range(n):
        k = int(rng.integers(20, len(vals) - 20))
        shifted = pd.Series(np.roll(vals, k), index=signal.index)
        nulls.append(perf_metrics(backtest(shifted, ret, cost_bps, short_funding_bps),
                                  ppy)["total_return"])
    nulls = np.asarray(nulls)
    return {"real_return": float(real),
            "null_mean": float(nulls.mean()), "null_p95": float(np.percentile(nulls, 95)),
            "p_value": float((nulls >= real).mean()), "n": n, "nulls": nulls}


def block_bootstrap_ci(strat_ret: pd.Series, block: int = 20, n: int = 2000,
                       seed: int = 42, ppy: int = 365) -> dict:
    """Block-bootstrap CIs for total return and Sharpe (preserves autocorrelation)."""
    rng = np.random.default_rng(seed)
    r = strat_ret.to_numpy()
    T = len(r)
    n_blocks = int(np.ceil(T / block))
    tot, shp = [], []
    for _ in range(n):
        starts = rng.integers(0, T - block, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:T]
        tot.append(np.prod(1 + sample) - 1)
        sd = sample.std()
        shp.append(sample.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0)
    tot, shp = np.asarray(tot), np.asarray(shp)
    return {
        "total_return_ci": (float(np.percentile(tot, 5)), float(np.percentile(tot, 95))),
        "sharpe_ci": (float(np.percentile(shp, 5)), float(np.percentile(shp, 95))),
        "p_return_negative": float((tot < 0).mean()),
    }


def shuffled_label_null(model_name: str, model_params: dict, strat: dict, X, fwd_returns,
                        ret, folds, feature_sets, cost_bps, short_funding_bps, mode,
                        n: int = 10, seed: int = 42, ppy: int = 365) -> dict:
    """Refit the winning ML config on label-shuffled data: OOS profit should die."""
    from .tuning import ml_eval_config
    rng = np.random.default_rng(seed)
    h = strat["horizon"]
    real = ml_eval_config(model_name, model_params, strat, X, fwd_returns, ret, folds,
                          feature_sets, cost_bps, short_funding_bps, mode, seed, ppy)
    nulls = []
    fr = fwd_returns[h]
    for i in range(n):
        perm = rng.permutation(len(fr))
        fr_shuf = pd.Series(fr.to_numpy()[perm], index=fr.index)
        res = ml_eval_config(model_name, model_params, strat, X, {h: fr_shuf}, ret, folds,
                             feature_sets, cost_bps, short_funding_bps, mode,
                             seed + i + 1, ppy)
        if np.isfinite(res["score"]):
            nulls.append(res["mean_return"])
    nulls = np.asarray(nulls)
    return {"real_mean_return": real["mean_return"],
            "null_mean": float(nulls.mean()) if len(nulls) else np.nan,
            "null_p95": float(np.percentile(nulls, 95)) if len(nulls) else np.nan,
            "p_value": float((nulls >= real["mean_return"]).mean()) if len(nulls) else np.nan,
            "n": int(len(nulls))}

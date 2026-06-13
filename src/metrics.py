"""Performance metrics, all computed on NET (post-cost) strategy returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def perf_metrics(bt: pd.DataFrame, periods_per_year: int = 365) -> dict:
    r = bt["strat_ret"]
    eq = bt["equity"]
    n = len(r)
    if n == 0:
        return {}
    total = float(eq.iloc[-1] - 1)
    years = n / periods_per_year
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else np.nan
    mu, sd = r.mean(), r.std()
    sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(mu / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else 0.0
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else np.nan
    in_mkt = bt["position"] != 0
    active = r[in_mkt]
    win_rate = float((active > 0).mean()) if len(active) else np.nan
    gains, losses = active[active > 0].sum(), -active[active < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else np.inf
    n_trades = int((bt["turnover"] > 0).sum())
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "exposure": float(in_mkt.mean()),
        "turnover_total": float(bt["turnover"].sum()),
        "cost_drag": float(bt["cost"].sum()),
        "n_trades": n_trades,
        "n_bars": n,
    }


def metrics_table(named_bts: dict[str, pd.DataFrame], periods_per_year: int = 365) -> pd.DataFrame:
    rows = {name: perf_metrics(bt, periods_per_year) for name, bt in named_bts.items()}
    cols = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
            "win_rate", "profit_factor", "exposure", "turnover_total", "cost_drag", "n_trades"]
    return pd.DataFrame(rows).T[cols]

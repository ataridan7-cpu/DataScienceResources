"""Matplotlib chart helpers for the research notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 110})


def equity_curves(named_bts: dict[str, pd.DataFrame], title: str = "Equity curves (net of costs)",
                  log: bool = True):
    fig, ax = plt.subplots()
    for name, bt in named_bts.items():
        lw = 2.5 if "BUY" in name.upper() or "B&H" in name.upper() else 1.4
        ax.plot(bt.index, bt["equity"], label=name, linewidth=lw)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("equity (growth of $1)")
    ax.legend(loc="best", fontsize=8)
    return fig


def drawdown(bt: pd.DataFrame, title: str = "Drawdown"):
    dd = bt["equity"] / bt["equity"].cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dd.index, dd, 0, alpha=0.5, color="firebrick")
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    return fig


def fold_returns_bar(fold_stats: list[dict], title: str = "Per-fold OOS net returns"):
    rets = [f["total_return"] for f in fold_stats]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["seagreen" if r > 0 else "firebrick" for r in rets]
    ax.bar(range(1, len(rets) + 1), rets, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel("net return")
    ax.set_title(title)
    return fig


def importance_bar(ranking: pd.Series, top: int = 20, title: str = "Permutation importance (OOS folds)"):
    r = ranking.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(r) + 1))
    ax.barh(r.index, r.values, color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def return_hist(strat_ret: pd.Series, title: str = "Strategy return distribution"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(strat_ret[strat_ret != 0], bins=60, alpha=0.8, color="steelblue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title)
    return fig


def rolling_sharpe(strat_ret: pd.Series, window: int = 90, ppy: int = 365,
                   title: str = "Rolling Sharpe"):
    rs = strat_ret.rolling(window).mean() / strat_ret.rolling(window).std() * np.sqrt(ppy)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(rs.index, rs)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{title} ({window}-bar)")
    return fig


def cost_sweep_plot(sweeps: dict[str, pd.DataFrame], title: str = "Cost sensitivity (per-side bps)"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, df in sweeps.items():
        ax.plot(df.index, df["total_return"], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("cost per side (bps)")
    ax.set_ylabel("net total return")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def null_distribution(nulls: np.ndarray, real: float, title: str = "Null test"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(nulls, bins=40, alpha=0.7, color="gray", label="null (shifted signals)")
    ax.axvline(real, color="seagreen", lw=2.5, label=f"real = {real:.1%}")
    ax.set_title(title)
    ax.legend()
    return fig

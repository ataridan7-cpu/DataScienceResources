"""Hand-rolled, strictly causal technical-indicator features.

Every feature value at bar t is computed exclusively from data up to and
including bar t (pandas rolling/ewm are causal; `center=True` is never used).
The no-look-ahead property is asserted by tests/test_no_lookahead.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- indicators
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (causal ewm smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    pctb = (close - lower) / (upper - lower)
    bw = (upper - lower) / mid
    return pctb, bw, upper, lower, mid


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo)
    return k, k.rolling(d).mean()


def efficiency_ratio(close: pd.Series, n: int = 10) -> pd.Series:
    """Kaufman ER: |net move| / path length. ~1 trending, ~0 chop."""
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)


def parkinson_vol(df: pd.DataFrame, n: int = 10) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


# ---------------------------------------------------------------- feature set
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (X, groups). X aligned to df.index; warmup rows contain NaN."""
    c, v = df["close"], df.get("volume")
    X = pd.DataFrame(index=df.index)
    groups: dict[str, list[str]] = {}

    def add(group: str, name: str, series: pd.Series):
        X[name] = series
        groups.setdefault(group, []).append(name)

    # --- returns / momentum ---
    for k in (1, 2, 3, 5):
        add("returns", f"ret_{k}", c.pct_change(k))
    add("returns", "logret_1", np.log(c / c.shift(1)))
    for k in (10, 21, 63):
        add("momentum", f"roc_{k}", c.pct_change(k))
    add("momentum", "er_10", efficiency_ratio(c, 10))
    add("momentum", "dist_high_63", c / df["high"].rolling(63).max() - 1)
    add("momentum", "dist_low_63", c / df["low"].rolling(63).min() - 1)

    # --- trend (stationary ratios, never raw MA levels) ---
    for n in (10, 20, 50, 100, 200):
        add("trend", f"sma_ratio_{n}", c / sma(c, n) - 1)
    for n in (12, 26):
        add("trend", f"ema_ratio_{n}", c / ema(c, n) - 1)
    for f, s in ((5, 20), (10, 50), (20, 100), (50, 200)):
        add("trend", f"macross_{f}_{s}", (sma(c, f) - sma(c, s)) / c)

    # --- oscillators ---
    for n in (7, 14, 21):
        add("oscillator", f"rsi_{n}", rsi(c, n) / 100)
    line, sig, hist = macd(c)
    add("oscillator", "macd_line", line / c)
    add("oscillator", "macd_signal", sig / c)
    add("oscillator", "macd_hist", hist / c)
    k_, d_ = stochastic(df)
    add("oscillator", "stoch_k", k_ / 100)
    add("oscillator", "stoch_d", d_ / 100)

    # --- volatility / range ---
    lr = np.log(c / c.shift(1))
    for n in (10, 21, 63):
        add("volatility", f"vol_{n}", lr.rolling(n).std())
    add("volatility", "vol_ratio_10_63", lr.rolling(10).std() / lr.rolling(63).std())
    add("volatility", "atr_pct", atr(df, 14) / c)
    pctb, bw, *_ = bollinger(c)
    add("volatility", "bb_pctb", pctb)
    add("volatility", "bb_bw", bw)
    add("volatility", "parkinson_10", parkinson_vol(df, 10))
    add("volatility", "hl_range", (df["high"] - df["low"]) / c)
    add("volatility", "co_ret", (c - df["open"]) / df["open"])

    # --- volume (optional) ---
    if v is not None and v.notna().any() and (v > 0).any():
        vz = (v - v.rolling(21).mean()) / v.rolling(21).std()
        add("volume", "vol_z_21", vz)
        add("volume", "vol_ratio_5_21", v.rolling(5).mean() / v.rolling(21).mean())
        add("volume", "logvol_chg", np.log(v.replace(0, np.nan)).diff())
        o = obv(df)
        add("volume", "obv_slope_10", (o - o.shift(10)) / v.rolling(10).mean().replace(0, np.nan))

    # --- calendar (cyclical) ---
    dow = df.index.dayofweek
    add("calendar", "dow_sin", pd.Series(np.sin(2 * np.pi * dow / 7), index=df.index))
    add("calendar", "dow_cos", pd.Series(np.cos(2 * np.pi * dow / 7), index=df.index))
    month = df.index.month
    add("calendar", "month_sin", pd.Series(np.sin(2 * np.pi * month / 12), index=df.index))
    add("calendar", "month_cos", pd.Series(np.cos(2 * np.pi * month / 12), index=df.index))

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, groups

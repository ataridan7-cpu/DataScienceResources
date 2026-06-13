"""Target/label builders. Forward shifts appear ONLY here (targets), never in features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Return from close_t to close_{t+horizon}. NaN for trailing rows."""
    return close.shift(-horizon) / close - 1


def sign_label(close: pd.Series, horizon: int = 1, band: float = 0.0) -> pd.Series:
    """Binary up/down label with optional neutral band.

    1 if fwd_ret > +band, 0 if fwd_ret < -band, NaN inside the band
    (banded samples are dropped from TRAINING only; prediction covers all rows).
    """
    fr = forward_return(close, horizon)
    y = pd.Series(np.nan, index=close.index)
    y[fr > band] = 1.0
    y[fr < -band] = 0.0
    return y


def triple_barrier(df: pd.DataFrame, horizon: int = 5, pt_mult: float = 2.0,
                   sl_mult: float = 1.0, atr_n: int = 14) -> pd.Series:
    """Lopez de Prado triple-barrier label (optional advanced target).

    +1 profit-take hit first, 0 stop-loss hit first, NaN if neither within
    `horizon` bars (time barrier -> sign of the period return).
    Barriers are ATR-scaled from the entry close.
    """
    from .features import atr  # local import avoids cycle
    a = atr(df, atr_n)
    close, high, low = df["close"], df["high"], df["low"]
    n = len(df)
    out = np.full(n, np.nan)
    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    av = a.to_numpy()
    for i in range(n - 1):
        if not np.isfinite(av[i]):
            continue
        upper = c[i] + pt_mult * av[i]
        lower = c[i] - sl_mult * av[i]
        end = min(i + horizon, n - 1)
        label = np.nan
        for j in range(i + 1, end + 1):
            hit_up, hit_dn = h[j] >= upper, l[j] <= lower
            if hit_up and hit_dn:   # both inside one bar -> ambiguous, drop
                label = np.nan
                break
            if hit_up:
                label = 1.0
                break
            if hit_dn:
                label = 0.0
                break
        else:  # time barrier
            label = 1.0 if c[end] > c[i] else 0.0
        out[i] = label
    return pd.Series(out, index=df.index)

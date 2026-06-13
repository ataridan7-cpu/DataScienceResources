"""Backtest accounting sanity: execution lag, fee charging, and B&H identity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import backtest, buy_and_hold, proba_to_signal  # noqa: E402
from src.metrics import perf_metrics  # noqa: E402


def _ret(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(0.001, 0.04, n), index=idx)


def test_always_long_zero_cost_equals_buy_hold():
    ret = _ret()
    sig = pd.Series(1.0, index=ret.index)
    bt = backtest(sig, ret, cost_bps=0.0)
    bh = buy_and_hold(ret, cost_bps=0.0)
    assert np.allclose(bt["equity"], bh["equity"])
    # zero-cost always-long compounds the asset exactly (after 1-bar entry lag)
    expected = (1 + ret).cumprod() / (1 + ret.iloc[0])
    assert np.allclose(bt["equity"].iloc[1:], expected.iloc[1:], rtol=1e-9)


def test_execution_lag_no_lookahead():
    # a signal that perfectly knows the current bar's sign must NOT earn it,
    # because the position is applied to the NEXT bar.
    ret = _ret()
    omniscient = (ret > 0).astype(float)  # same-bar knowledge
    bt = backtest(omniscient, ret, cost_bps=0.0)
    # position is shifted, so realized return correlates with lagged signal only
    realized = bt["position"] * ret
    assert np.allclose(realized.values, omniscient.shift(1).fillna(0).values * ret.values)


def test_costs_charged_on_turnover():
    ret = _ret()
    # flip every bar -> turnover ~1 each step -> heavy cost drag
    flip = pd.Series(([1.0, -1.0] * 200)[:len(ret)], index=ret.index)
    cheap = perf_metrics(backtest(flip, ret, cost_bps=0.0))
    pricey = perf_metrics(backtest(flip, ret, cost_bps=50.0))
    assert pricey["cost_drag"] > cheap["cost_drag"]
    assert pricey["total_return"] < cheap["total_return"]


def test_proba_to_signal_modes():
    p = pd.Series([0.2, 0.5, 0.8], index=range(3))
    lo = proba_to_signal(p, mode="long_only", sizing="binary", thr_long=0.55)
    assert set(lo.unique()) <= {0.0, 1.0} and lo.iloc[2] == 1.0
    ls = proba_to_signal(p, mode="long_short", sizing="binary", thr_long=0.55, thr_short=0.45)
    assert ls.iloc[0] == -1.0 and ls.iloc[2] == 1.0
    sc = proba_to_signal(p, mode="long_only", sizing="scaled", scale=0.2)
    assert (sc >= 0).all() and (sc <= 1).all()


if __name__ == "__main__":
    for fn in [test_always_long_zero_cost_equals_buy_hold, test_execution_lag_no_lookahead,
               test_costs_charged_on_turnover, test_proba_to_signal_modes]:
        fn()
    print("backtest tests passed")

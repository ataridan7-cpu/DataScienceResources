"""Assert features are strictly causal: a feature value at bar t must not change
when future bars are removed. This is the core no-look-ahead guarantee."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import load_prices  # noqa: E402
from src.features import build_features  # noqa: E402
from src.config import CFG  # noqa: E402


def _toy_df(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    high = price * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": price, "high": high, "low": low, "close": price,
                         "volume": rng.uniform(1e6, 5e6, n)}, index=idx).rename_axis("timestamp")


def test_features_are_causal():
    df = _toy_df()
    X_full, _ = build_features(df)
    # truncating the future must not change past feature values
    for cut in (450, 550, 650):
        X_trunc, _ = build_features(df.iloc[:cut])
        common = X_full.iloc[:cut].dropna().index.intersection(X_trunc.dropna().index)
        assert len(common) > 50, f"too few comparable rows at cut={cut}"
        diff = (X_full.loc[common] - X_trunc.loc[common]).abs().max().max()
        assert diff < 1e-8, f"look-ahead detected at cut={cut}: max diff {diff}"


def test_real_data_features_causal():
    if not CFG.raw_csv.exists():
        return
    df = load_prices(CFG.raw_csv)
    X_full, _ = build_features(df)
    cut = int(len(df) * 0.7)
    X_trunc, _ = build_features(df.iloc[:cut])
    common = X_full.iloc[:cut].dropna().index.intersection(X_trunc.dropna().index)[30:]
    diff = (X_full.loc[common] - X_trunc.loc[common]).abs().max().max()
    assert diff < 1e-6, f"look-ahead on real data: {diff}"


if __name__ == "__main__":
    test_features_are_causal()
    test_real_data_features_causal()
    print("no-look-ahead tests passed")

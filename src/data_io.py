"""Data loading, schema normalization, cleaning and granularity detection."""
from __future__ import annotations

import pandas as pd

# Common aliases -> canonical column names
_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "datetime", "day", "open time", "open_time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj close", "adj_close", "price"],
    "volume": ["volume", "v", "vol"],
    "market_cap": ["market cap", "market_cap", "marketcap", "mcap"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lower = {c: c.strip().lower() for c in df.columns}
    for canon, aliases in _ALIASES.items():
        for col, low in lower.items():
            if low in aliases and canon not in rename.values():
                rename[col] = canon
    df = df.rename(columns=rename)
    missing = {"timestamp", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing after alias mapping: {missing}. "
                         f"Got columns: {list(df.columns)}")
    return df


def load_prices(csv_path, dayfirst: bool = True) -> pd.DataFrame:
    """Load OHLCV csv -> cleaned ascending UTC-indexed DataFrame."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):  # epoch s or ms
        unit = "ms" if ts.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, dayfirst=dayfirst, utc=True, format="mixed")
    df = df.drop(columns=["timestamp"]).set_index(idx).rename_axis("timestamp")

    df = df[~df.index.duplicated(keep="first")].sort_index()

    # numeric coercion + basic sanity
    for c in [c for c in ["open", "high", "low", "close", "volume", "market_cap"] if c in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # OHLC consistency: clip high/low to envelope rather than dropping rows
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    assert df.index.is_monotonic_increasing
    return df


def detect_granularity(df: pd.DataFrame) -> dict:
    """Modal bar spacing -> frequency info used to derive annualization & windows."""
    diffs = df.index.to_series().diff().dropna()
    modal = diffs.mode().iloc[0]
    bar_seconds = modal.total_seconds()
    periods_per_year = int(round(365 * 24 * 3600 / bar_seconds))
    gaps = diffs[diffs != modal]
    return {
        "bar": modal,
        "bar_seconds": bar_seconds,
        "periods_per_year": periods_per_year,
        "n_rows": len(df),
        "start": df.index[0],
        "end": df.index[-1],
        "n_gaps": int(len(gaps)),
        "gap_examples": gaps.head(5),
    }


def load_btc_data(cache_path: Path) -> "pd.DataFrame | None":
    """Download BTC-USD daily OHLCV via yfinance, cache to parquet.

    Returns None gracefully if yfinance is unavailable or download fails.
    BTC features in build_features() are simply omitted when this returns None.
    """
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", start="2019-01-01", progress=False, auto_adjust=True)
        if btc is None or len(btc) == 0:
            return None
        # flatten MultiIndex columns produced by recent yfinance versions
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [c[0].lower() for c in btc.columns]
        else:
            btc.columns = [c.lower() for c in btc.columns]
        btc.index = pd.to_datetime(btc.index, utc=True)
        btc = btc.rename_axis("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in btc.columns:
                return None
        btc = btc[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        btc.to_parquet(cache_path)
        return btc
    except Exception:
        return None


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    info = detect_granularity(df)
    return pd.DataFrame({
        "value": {
            "rows": info["n_rows"],
            "start": str(info["start"].date()),
            "end": str(info["end"].date()),
            "bar": str(info["bar"]),
            "periods_per_year": info["periods_per_year"],
            "irregular_gaps": info["n_gaps"],
            "first_close": round(float(df['close'].iloc[0]), 4),
            "last_close": round(float(df['close'].iloc[-1]), 4),
        }
    })

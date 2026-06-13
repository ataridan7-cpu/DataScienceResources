# SOL Trading-Strategy Research Pipeline

An end-to-end, **leakage-controlled** research pipeline that engineers features,
searches exhaustively over feature combinations / rule parameters / labels /
models / hyperparameters, and builds trading strategies on **Solana (SOL)** daily
data — maximizing **net (post-cost) profit** and comparing honestly against
**buy & hold**.

Dataset: [`craigdagama/solana-historical-data`](https://www.kaggle.com/datasets/craigdagama/solana-historical-data)
(daily OHLCV, ~1.9k bars, 2020‑04 → 2025‑06). Already vendored under `data/raw/`.

## Quick start

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch

pytest tests/ -q                 # no-look-ahead + backtest-accounting tests
python run_pipeline.py           # runs/caches all search stages -> results/
python build_notebook.py         # (re)generate the notebook
jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_sol_strategy_research.ipynb
```

The notebook `notebooks/01_sol_strategy_research.ipynb` is the narrative
deliverable; it also runs unchanged in **Google Colab** (cell 1 downloads the
dataset via `kagglehub`). Live numbers are written to `results/summary.json`.

## How profit is maximized — *honestly*

Maximizing profit on historical data is trivial if you cheat. Every stage is
built to make the reported number a defensible estimate of forward performance:

| Concern | Defense | Where |
|---|---|---|
| Look-ahead in features | strictly causal, hand-rolled indicators; unit test recomputes on truncated history | `src/features.py`, `tests/test_no_lookahead.py` |
| Same-bar execution | decide at close *t*, trade at *t+1* (one-bar lag) | `src/backtest.py` |
| CV leakage | purged + embargoed **walk-forward** | `src/cv.py` |
| Free trading | fees + slippage on every position change; short funding haircut | `src/backtest.py`, `src/config.py` |
| Tuning to accuracy ≠ profit | Optuna objective = **mean OOS net return − 0.25·std** through the full backtest | `src/tuning.py` |
| Selection leakage | feature search nested inside training folds only | `src/selection.py` |
| Data snooping | **holdout touched once**; circular-shift null; shuffled-label null; block-bootstrap CIs | `src/robustness.py`, `src/pipeline.py` |

## What it searches

- **48 causal features** — returns/momentum, trend (price-ratio MAs), oscillators
  (RSI, MACD, stochastic), volatility (realized vol, ATR, Bollinger, Parkinson),
  volume, cyclical calendar — `src/features.py`.
- **Feature combinations** — correlation prune → permutation importance →
  greedy forward selection by net profit → named sets (`top10/15/25`, `greedy`).
- **Rule strategies (exhaustive grid)** — MA-cross, MACD, RSI, Bollinger,
  Donchian breakout, in long-only and long/short.
- **10 ML families (Optuna)** — logistic, SVM-RBF, kNN, RandomForest, XGBoost,
  LightGBM, CatBoost, and compact CPU **MLP / LSTM / Transformer**.
- **Strategy wrapper** — label horizon {1,2,3,5}, neutral band, binary vs
  confidence-scaled sizing, threshold — searched jointly with model params.
- **Modes** — long-only (hold/cash) and long/short, each judged separately.

## Layout

```
src/         config, data_io, features, labels, cv, models, tuning,
             selection, backtest, metrics, plots, robustness, pipeline
tests/       test_no_lookahead.py, test_backtest.py
data/raw/    vendored Kaggle CSV
results/     tuning/ caches + summary.json (regenerated)
notebooks/   01_sol_strategy_research.ipynb   (the deliverable)
run_pipeline.py, build_notebook.py
```

## Reading the result

Buy & hold of SOL since 2020 is an extraordinarily strong benchmark, so the
realistic, defensible win is usually **better risk-adjusted return** (higher
Sharpe / Calmar, shallower drawdown) by sitting out crashes — not necessarily a
higher absolute return. The notebook reports both and states the verdict per
mode from the untouched holdout.

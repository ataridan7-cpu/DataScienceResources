"""Construct notebooks/01_sol_strategy_research.ipynb programmatically."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

nb = new_notebook()
cells = []


def md(text): cells.append(new_markdown_cell(text.strip("\n")))
def code(text): cells.append(new_code_cell(text.strip("\n")))


# ======================================================= TITLE
md(r"""
# Solana (SOL) — Profit-Maximizing Trading Strategy Research

**Goal.** Engineer features, search exhaustively over feature combinations, rule
parameters, labels, models and hyperparameters, and build trading strategies that
**maximize net (post-cost) profit** — then compare honestly against **buy & hold**.

| Trap | Defense |
|---|---|
| Look-ahead in features | Strictly causal indicators; `tests/test_no_lookahead.py` |
| Same-bar execution | Decide at close *t*, trade at *t+1* (one-bar lag) |
| CV leakage | `PurgedWalkForward` with purge + embargo |
| Free trading | Fees + slippage on every position change |
| Data snooping | Holdout touched once; null tests; block-bootstrap CIs |

**Pipeline:** load → EDA → 48 causal features → feature analysis → labels + CV →
rule grids → feature-combination search → 10 ML families × 2 modes (Optuna, profit
objective) → one-shot holdout → robustness → verdict.
""")

# ======================================================= 1. ALL IMPORTS
md("## 1. Imports & configuration")

code(r"""
# ── standard library ─────────────────────────────────────────────────────────
import sys, os, json, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# ── scientific stack ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, acf

# ── pipeline modules ─────────────────────────────────────────────────────────
from src.config import CFG
from src.data_io import load_prices, summarize, detect_granularity
from src.features import build_features
from src.labels import forward_return, sign_label
from src.cv import PurgedWalkForward, dev_holdout_split
from src.backtest import backtest, buy_and_hold, proba_to_signal
from src.metrics import perf_metrics, metrics_table
from src.models import ML_MODELS, RULES
from src.pipeline import (load_dataset, stage_rules, stage_feature_sets,
                           stage_ml, final_leaderboard)
from src.robustness import cost_sweep, circular_shift_null, block_bootstrap_ci
from src import plots

# ── display settings ─────────────────────────────────────────────────────────
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
pd.set_option("display.max_columns", 20)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10})
CMAP_DIV = LinearSegmentedColormap.from_list("rg", ["#d73027","#f7f7f7","#1a9850"])
CMAP_SEQ = "YlOrRd"
print("All imports OK")
""")

# ======================================================= 2. DATA DOWNLOAD
md(r"""
## 2. Data download & load

The original Kaggle snippet works unchanged in Colab. In this repo the CSV is
already vendored under `data/raw/`.
""")

code(r"""
# ── Original Kaggle download (works in Colab as-is) ──────────────────────────
# import kagglehub
# path = kagglehub.dataset_download("craigdagama/solana-historical-data")
# print("Path to dataset files:", path)

def resolve_csv():
    if CFG.raw_csv.exists():
        return CFG.raw_csv
    import kagglehub
    path = kagglehub.dataset_download("craigdagama/solana-historical-data")
    return Path(sorted(glob.glob(os.path.join(path, "*.csv")))[0])

csv_path = resolve_csv()
df = load_prices(csv_path)
info = detect_granularity(df)
print(summarize(df).to_string())
print(f"\nIrregular gaps : {info['n_gaps']}")
print(f"Periods / year : {info['periods_per_year']}")
df.tail(3)
""")

# ======================================================= 3. EDA CHARTS
md("## 3. Exploratory data analysis")

code(r"""
# ── 3a. Price + volume overview ───────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# price (log)
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(df.index, df["close"], lw=1.2, color="steelblue")
ax0.set_yscale("log")
ax0.set_title("SOL close price — log scale (full history)", fontweight="bold")
ax0.set_ylabel("USD (log)")
ax0.xaxis.set_major_locator(mdates.YearLocator())
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# volume
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(df.index, df["volume"], width=1, color="slategray", alpha=0.7)
ax1.set_title("Volume")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v/1e9:.1f}B"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# daily returns
ret_all = df["close"].pct_change().dropna()
ax2 = fig.add_subplot(gs[1, 1])
ax2.hist(ret_all, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
ax2.axvline(0, color="k", lw=0.8)
ax2.set_title("Daily return distribution")
ax2.set_xlabel("return")

# rolling vol
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(df.index[1:], ret_all.rolling(30).std().iloc[:] * np.sqrt(365),
         color="darkorange", lw=1.1)
ax3.set_title("30-day rolling annualised vol")
ax3.set_ylabel("annualised σ")
ax3.xaxis.set_major_locator(mdates.YearLocator())

# drawdown from ATH
prices = df["close"]
dd = prices / prices.cummax() - 1
ax4 = fig.add_subplot(gs[2, 1])
ax4.fill_between(df.index, dd, 0, color="firebrick", alpha=0.55)
ax4.set_title("Drawdown from all-time high")
ax4.set_ylabel("drawdown")
ax4.xaxis.set_major_locator(mdates.YearLocator())

fig.suptitle("SOL — Full History Overview", fontsize=13, fontweight="bold", y=1.01)
plt.show()

bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
print(f"Buy & hold over full history: {bh:,.0%}  "
      f"(${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f})")
print("==> This is the benchmark every strategy must beat.")
""")

code(r"""
# ── 3b. Monthly returns heatmap ───────────────────────────────────────────────
monthly = (ret_all + 1).resample("ME").prod() - 1
pivot = monthly.to_frame("ret")
pivot["year"] = pivot.index.year
pivot["month"] = pivot.index.month
hm = pivot.pivot(index="year", columns="month", values="ret")
hm.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 4.5))
im = ax.imshow(hm.values, cmap=CMAP_DIV, aspect="auto",
               vmin=-0.6, vmax=0.6)
ax.set_xticks(range(12)); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index)
for i in range(len(hm)):
    for j in range(12):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.0%}", ha="center", va="center",
                    fontsize=7.5, color="black" if abs(v) < 0.35 else "white")
plt.colorbar(im, ax=ax, label="monthly return", fraction=0.03)
ax.set_title("Monthly returns heatmap (green=up, red=down)", fontweight="bold")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── 3c. Stationarity & autocorrelation ───────────────────────────────────────
p_price = adfuller(df["close"].dropna())[1]
p_ret   = adfuller(ret_all)[1]
print(f"ADF p-value — price level : {p_price:.4f}  (>0.05 → non-stationary, as expected)")
print(f"ADF p-value — daily return: {p_ret:.2e}  (<0.05 → stationary)")
print("Features are expressed as returns / price-ratios (stationary) — never raw levels.")

acf_vals = acf(ret_all, nlags=30, fft=True)[1:]
conf = 1.96 / np.sqrt(len(ret_all))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].bar(range(1, 31), acf_vals, color=["steelblue" if abs(v) > conf else "lightblue"
                                            for v in acf_vals])
axes[0].axhline(conf, color="red", ls="--", lw=0.9, label=f"95% CI (±{conf:.3f})")
axes[0].axhline(-conf, color="red", ls="--", lw=0.9)
axes[0].axhline(0, color="k", lw=0.7)
axes[0].set_title("Autocorrelation of daily returns (lags 1–30)")
axes[0].set_xlabel("lag (days)"); axes[0].legend()

axes[1].scatter(ret_all.iloc[:-1].values, ret_all.iloc[1:].values,
                alpha=0.15, s=6, color="steelblue")
axes[1].set_xlabel("return(t)"); axes[1].set_ylabel("return(t+1)")
axes[1].set_title("Return(t) vs Return(t+1) scatter")
z = np.polyfit(ret_all.iloc[:-1], ret_all.iloc[1:], 1)
xx = np.linspace(ret_all.min(), ret_all.max(), 100)
axes[1].plot(xx, np.polyval(z, xx), color="firebrick", lw=1.5,
             label=f"slope={z[0]:.3f}")
axes[1].legend()
plt.tight_layout(); plt.show()
""")

# ======================================================= 4. FEATURE ENGINEERING
md(r"""
## 4. Feature engineering (48 strictly-causal features)

Every indicator is hand-rolled (no TA-Lib) so causality is **guaranteed and
unit-tested**. Trend features are price-ratios (stationary), never raw MA levels.
""")

code(r"""
data = load_dataset(CFG)
X, groups = data["X"], data["groups"]
print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features  "
      f"(warmup NaN rows dropped from {len(df)})")
for g, cols in groups.items():
    print(f"  {g:11s}: {len(cols):2d}  →  {', '.join(cols[:6])}{'...' if len(cols)>6 else ''}")
""")

# ======================================================= 5. FEATURE ANALYSIS
md(r"""
## 5. Feature analysis

Before any modelling: distributions, the full **correlation heatmap**, predictive
signal vs next-day return, and autocorrelation structure.
""")

code(r"""
# ── 5a. Feature distributions by group (box plot) ────────────────────────────
Xn = X.copy()
for col in Xn.columns:
    Xn[col] = (Xn[col] - Xn[col].mean()) / (Xn[col].std() + 1e-12)

group_list = list(groups.items())
n_groups = len(group_list)
fig, axes = plt.subplots(1, n_groups, figsize=(16, 4.5))
for ax, (g, cols) in zip(axes, group_list):
    data_vals = [Xn[c].dropna().values for c in cols]
    bp = ax.boxplot(data_vals, vert=True, patch_artist=True,
                    medianprops=dict(color="firebrick", lw=1.5),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(cols)+1))
    ax.set_xticklabels([c.replace("_"," ") for c in cols], rotation=70, ha="right", fontsize=7)
    ax.set_title(g, fontweight="bold", fontsize=9)
    ax.set_ylim(-5, 5)
fig.suptitle("z-scored feature distributions by group", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── 5b. Full feature correlation heatmap ─────────────────────────────────────
corr = X.corr()
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(15, 13))
im = ax.imshow(corr.values, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=6.5)
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.025, pad=0.01)
ax.set_title("Full feature–feature correlation matrix (48 × 48)", fontsize=13, fontweight="bold")

# draw group separators
boundaries, pos = 0, []
for g, cols in groups.items():
    boundaries += len(cols)
    pos.append((boundaries - len(cols)/2, g))
    ax.axhline(boundaries - 0.5, color="k", lw=0.8)
    ax.axvline(boundaries - 0.5, color="k", lw=0.8)
for p, label in pos:
    ax.text(len(corr)+0.3, p, label, va="center", fontsize=7.5, color="navy")

plt.tight_layout(); plt.show()

# most correlated pairs (outside same feature)
tri = corr.where(np.tril(np.ones_like(corr, dtype=bool), k=-1))
pairs = (tri.stack().abs()
            .sort_values(ascending=False)
            .rename("abs_corr").reset_index())
pairs.columns = ["feat_a","feat_b","abs_corr"]
print("Top 10 most correlated feature pairs:")
print(pairs.head(10).to_string(index=False))
""")

code(r"""
# ── 5c. Feature correlation with next-day return (predictive signal) ──────────
fwd1 = data["fwd"][1]
feat_ret_corr = pd.Series({
    col: fwd1.corr(X[col]) for col in X.columns
}).sort_values(key=abs, ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# bar chart — top 20 by absolute correlation
top20 = feat_ret_corr.head(20)[::-1]
colors = ["seagreen" if v > 0 else "firebrick" for v in top20.values]
axes[0].barh(top20.index, top20.values, color=colors)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_title("Feature correlation with next-day return\n(top 20 by |r|)", fontweight="bold")
axes[0].set_xlabel("Pearson r")

# full heatmap: feature × next-day/week/month returns
horizons = {f"fwd_{h}d": data["fwd"][h] for h in (1, 2, 3, 5)}
hm_data = pd.DataFrame({
    col: {hname: horizons[hname].corr(X[col]) for hname in horizons}
    for col in X.columns
}).T
hm_sorted = hm_data.reindex(feat_ret_corr.index)
im = axes[1].imshow(hm_sorted.values.T, cmap=CMAP_DIV, vmin=-0.15, vmax=0.15, aspect="auto")
axes[1].set_xticks(range(len(hm_sorted))); axes[1].set_xticklabels(hm_sorted.index, rotation=90, fontsize=5.5)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(list(horizons.keys()))
plt.colorbar(im, ax=axes[1], label="Pearson r", fraction=0.04)
axes[1].set_title("Feature vs forward-return\ncorrelation (by horizon)", fontweight="bold")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── 5d. Top features: scatter + rolling correlation with next-day return ───────
top4_feats = list(feat_ret_corr.abs().sort_values(ascending=False).head(4).index)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for col, ax_sc, ax_rl in zip(top4_feats, axes[0], axes[1]):
    feat = X[col].dropna()
    aligned = fwd1.reindex(feat.index).dropna()
    feat_a = feat.reindex(aligned.index)
    r = stats.pearsonr(feat_a, aligned)[0]
    ax_sc.scatter(feat_a.values, aligned.values, alpha=0.15, s=5, color="steelblue")
    z = np.polyfit(feat_a.values, aligned.values, 1)
    xx = np.linspace(feat_a.min(), feat_a.max(), 80)
    ax_sc.plot(xx, np.polyval(z, xx), color="firebrick", lw=1.5)
    ax_sc.set_title(f"{col}\nr={r:.3f}", fontsize=8, fontweight="bold")
    ax_sc.set_xlabel("feature", fontsize=7); ax_sc.set_ylabel("fwd return", fontsize=7)
    roll_r = feat_a.rolling(90).corr(aligned)
    ax_rl.plot(roll_r.index, roll_r.values, lw=1.1, color="darkorange")
    ax_rl.axhline(0, color="k", lw=0.7)
    ax_rl.fill_between(roll_r.index, roll_r, 0,
                       where=roll_r > 0, alpha=0.25, color="seagreen")
    ax_rl.fill_between(roll_r.index, roll_r, 0,
                       where=roll_r < 0, alpha=0.25, color="firebrick")
    ax_rl.set_title(f"90-day rolling r(t)", fontsize=8)
    ax_rl.set_ylim(-0.5, 0.5)
    ax_rl.xaxis.set_major_locator(mdates.YearLocator())
    ax_rl.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Top-4 predictive features: scatter vs fwd return (top row) "
             "& rolling correlation (bottom row)", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── 5e. Feature value by up vs down next-day move ────────────────────────────
y_dir = (fwd1 > 0).astype(int).reindex(X.index)

fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
axes = axes.flatten()
top10 = list(feat_ret_corr.abs().sort_values(ascending=False).head(10).index)
for ax, col in zip(axes, top10):
    up = X[col][y_dir == 1].dropna()
    dn = X[col][y_dir == 0].dropna()
    vmin = np.percentile(pd.concat([up, dn]), 1)
    vmax = np.percentile(pd.concat([up, dn]), 99)
    bins = np.linspace(vmin, vmax, 35)
    ax.hist(up, bins=bins, alpha=0.55, color="seagreen", density=True, label="up")
    ax.hist(dn, bins=bins, alpha=0.55, color="firebrick", density=True, label="down")
    ax.set_title(col, fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
fig.suptitle("Feature distribution conditioned on next-day direction "
             "(green=up day, red=down day)", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ======================================================= 6. LABELS + CV
md(r"""
## 6. Labels & purged walk-forward cross-validation

**Default label:** sign of next-bar return (with cost-aware neutral band).
Horizons {1,2,3,5} are all searched by the Optuna studies.
""")

code(r"""
# ── CV fold layout ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3.2))
fold_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(data["folds"])))
for i, (tr, va) in enumerate(data["folds"]):
    ax.barh(i, len(tr), color=fold_colors[i], edgecolor="none")
    ax.barh(i, len(va), left=tr[-1]+1+CFG.embargo_bars,
            color="darkorange", edgecolor="none", alpha=0.85)
ax.barh(len(data["folds"]), len(data["hold_idx"]),
        left=len(data["dev_idx"]), color="firebrick", alpha=0.85)
ax.set_yticks(range(len(data["folds"])+1))
ax.set_yticklabels([f"fold {i+1}" for i in range(len(data["folds"]))] + ["HOLDOUT"])
ax.set_xlabel("row index (each row = 1 trading day)")
ax.set_title("Walk-forward layout — blue=train, orange=test, red=final holdout (1×)",
             fontweight="bold")

# annotate sizes
for i, (tr, va) in enumerate(data["folds"]):
    ax.text(len(tr)/2, i, f"{len(tr)}", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")
    ax.text(tr[-1]+1+CFG.embargo_bars + len(va)/2, i, f"{len(va)}",
            ha="center", va="center", fontsize=7.5, color="black", fontweight="bold")
hold_mid = len(data["dev_idx"]) + len(data["hold_idx"])/2
ax.text(hold_mid, len(data["folds"]), f"{len(data['hold_idx'])}",
        ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
plt.tight_layout(); plt.show()

print(f"{len(data['folds'])} folds | "
      f"dev={len(data['dev_idx'])} rows | holdout={len(data['hold_idx'])} rows | "
      f"purge={CFG.embargo_bars} bars")
""")

# ======================================================= 7. RULE BASELINES
md(r"""
## 7. Baselines — buy & hold + exhaustive rule grids

Five classic technical strategies (MA-cross, MACD, RSI, Bollinger, Donchian)
exhaustively grid-searched over all parameter combinations, scored by **mean
OOS net fold return**. Sets the bar before any ML.
""")

code(r"""
rules = stage_rules(data, CFG)
show = ["rule", "params", "mean_excess_return", "mean_return", "mean_sharpe",
        "pct_folds_beat_bh", "score"]
print("Rule grids ranked by EXCESS return vs B&H (score = mean_excess − 0.25·std):\n")
for mode in ("long_only", "long_short"):
    print(f"### {mode} — top 5:")
    print(rules[mode][show].head(5).to_string(index=False), "\n")
""")

code(r"""
# ── Rule leaderboard visualised — EXCESS return vs B&H ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (mode, rlb) in zip(axes, rules.items()):
    top = rlb.head(15).copy()
    top["label"] = top["rule"] + " " + top["params"].str[:24]
    colors = ["seagreen" if e > 0 else "firebrick" for e in top["mean_excess_return"]]
    ax.barh(range(len(top)), top["mean_excess_return"], color=colors,
            edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["label"], fontsize=6.5)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel("mean OOS excess return vs B&H (dev folds)")
    ax.set_title(f"Top-15 rule configs — {mode}\n(green = beats buy & hold)",
                 fontweight="bold", fontsize=9)
plt.tight_layout(); plt.show()
""")

# ======================================================= 8. FEATURE SELECTION
md(r"""
## 8. Feature-combination search

`src/selection.py`: correlation-prune (|ρ|>0.95) → permutation importance →
greedy forward selection by OOS profit. Named sets fed to Optuna studies.
""")

code(r"""
fsets = stage_feature_sets(data, CFG)
ranking = pd.Series(fsets["ranking"]).sort_values(ascending=False)
print("Candidate feature sets:", {k: len(v) for k, v in fsets["sets"].items()})
print("Greedy (profit-selected) set:", fsets["sets"]["greedy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# permutation importance bar
top20 = ranking.head(20)[::-1]
axes[0].barh(top20.index, top20.values, color="steelblue", alpha=0.8)
axes[0].set_title("Permutation importance\n(OOS folds, top 20)", fontweight="bold")
axes[0].set_xlabel("mean permutation importance")

# feature-set Venn-like membership table
feature_set_df = pd.DataFrame(
    {k: {f: (f in v) for f in ranking.index} for k, v in fsets["sets"].items()},
).astype(int)
membership = feature_set_df.T  # sets as rows
im = axes[1].imshow(membership.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
axes[1].set_xticks(range(len(membership.columns)))
axes[1].set_xticklabels(membership.columns, rotation=90, fontsize=5.5)
axes[1].set_yticks(range(len(membership.index)))
axes[1].set_yticklabels(membership.index, fontsize=8)
axes[1].set_title("Feature membership by set\n(blue = included)", fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ======================================================= 9. ML SEARCH
md(r"""
## 9. Model search — 10 families × 2 modes, profit-maximizing Optuna

Each study jointly searches model hyperparameters and strategy wrapper (horizon,
threshold, sizing, feature set), **objective = mean OOS _excess return vs buy & hold_
− 0.25·std** through the full cost-aware backtest. Optimising *alpha over B&H*, not
raw return, so a strategy is only rewarded for **beating the benchmark**. ML uses
Optuna TPE + median pruning.
""")

code(r"""
ml_results = stage_ml(data, fsets, ML_MODELS, CFG)

rows = []
for key, r in ml_results.items():
    if not r.get("best_params"): continue
    name, mode = key.split("__")
    a = r.get("best_attrs") or {}
    rows.append({"model": name, "mode": mode,
                 "dev_score_excess":  round(r["best_score"], 4),
                 "dev_mean_excess":   round(a.get("mean_excess_return", np.nan), 4),
                 "dev_mean_return":   round(a.get("mean_return", np.nan), 4),
                 "dev_mean_sharpe":   round(a.get("mean_sharpe", np.nan), 3),
                 "folds_beat_bh":     a.get("pct_folds_beat_bh"),
                 "horizon":           r["best_params"].get("horizon"),
                 "feature_set":       r["best_params"].get("feature_set"),
                 "sizing":            r["best_params"].get("sizing")})
dev_lb = (pd.DataFrame(rows)
          .sort_values("dev_score_excess", ascending=False)
          .reset_index(drop=True))
print("Dev-set leaderboard — ranked by mean OOS EXCESS return vs B&H (− 0.25·std):")
dev_lb
""")

code(r"""
# ── Dev leaderboard bar chart — EXCESS return vs B&H ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mode in zip(axes, ("long_only", "long_short")):
    sub = dev_lb[dev_lb["mode"] == mode].sort_values("dev_mean_excess",
                                                      ascending=True).tail(10)
    colors = ["seagreen" if e > 0 else "firebrick" for e in sub["dev_mean_excess"]]
    ax.barh(range(len(sub)), sub["dev_mean_excess"], color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub["model"], fontsize=8)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel("dev mean OOS excess return vs B&H")
    ax.set_title(f"Dev mean EXCESS return (alpha) — {mode}\n"
                 f"(>0 = beats buy & hold)", fontweight="bold", fontsize=9)
    for i, (_, row) in enumerate(sub.iterrows()):
        fb = row["folds_beat_bh"]
        ax.text(row["dev_mean_excess"] + (0.01 if row["dev_mean_excess"] >= 0 else -0.01), i,
                f"{fb:.0%} beat B&H", va="center", fontsize=6.5,
                ha="left" if row["dev_mean_excess"] >= 0 else "right")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── Walk-forward fold stability for best ML model ────────────────────────────
best_ml_key = dev_lb.iloc[0]
best_model_name = best_ml_key["model"]
best_mode = best_ml_key["mode"]
res_key = f"{best_model_name}__{best_mode}"
best_res = ml_results[res_key]
fold_stats = best_res.get("fold_stats") or (best_res.get("top5") or [{}])[0]
# top5 have fold_stats nested:
fold_stats = (best_res.get("top5") or [{}])[0]
all_fold_rets = []
for tk in best_res.get("top5", []):
    all_fold_rets.append([f["total_return"] for f in (tk.get("fold_stats") or [])])

if all_fold_rets and all_fold_rets[0]:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(all_fold_rets[0]))
    for i, fr in enumerate(all_fold_rets[:5]):
        ax.plot(x + 1, fr, marker="o", lw=1.3,
                label=f"top{i+1} config (mean={np.mean(fr):.1%})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("walk-forward fold"); ax.set_ylabel("net return")
    ax.set_title(f"Fold-by-fold returns for best {best_model_name} configs\n"
                 f"({best_mode}) — stability check", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("(fold-level stats not stored for this model — skipping stability chart)")
""")

# ======================================================= 10. HOLDOUT EVAL
md(r"""
## 10. The moment of truth — one-shot holdout evaluation

Every tuned config refit on the **full dev set** and evaluated **once** on the
untouched holdout (most recent 15%), net of costs. Nothing here was optimised
against this window.
""")

code(r"""
lb, signals = final_leaderboard(data, ml_results, rules, fsets, CFG, top_rules=3)
bh_row = lb[lb.strategy == "BUY & HOLD"].iloc[0]
hold_ret = data["ret"].iloc[data["hold_idx"]]

print(f"HOLDOUT buy & hold: return={bh_row['total_return']:.1%}  "
      f"Sharpe={bh_row['sharpe']:.2f}  maxDD={bh_row['max_drawdown']:.1%}")
print("All strategies below are scored by EXCESS RETURN over this benchmark.\n")

# excess_return & vs_bh_x come straight from the pipeline (profit vs B&H, not alone)
cols = ["strategy","type","mode","total_return","excess_return","vs_bh_x",
        "sharpe","max_drawdown","calmar","win_rate","n_trades"]
print("Holdout leaderboard (ranked by EXCESS return vs Buy & Hold):")
lb[cols].head(15)
""")

code(r"""
# ── Equity curves — top strategies vs B&H ────────────────────────────────────
def _bt(key, sig):
    funding = CFG.short_funding_bps_per_bar if key.endswith("long_short") else 0.0
    return backtest(sig, hold_ret, CFG.cost_bps_per_side, funding)

ranked_keys = sorted(signals.items(),
                     key=lambda kv: _bt(*kv)["equity"].iloc[-1], reverse=True)

named = {"BUY & HOLD": buy_and_hold(hold_ret, CFG.cost_bps_per_side)}
for key, sig in ranked_keys[:5]:
    named[key.replace("__", " | ")] = _bt(key, sig)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
# log equity
for name, bt_df in named.items():
    lw = 2.5 if "BUY" in name else 1.4
    ls = "--" if "BUY" in name else "-"
    axes[0].plot(bt_df.index, bt_df["equity"], label=name, lw=lw, ls=ls)
axes[0].set_yscale("log"); axes[0].set_ylabel("equity (log, $1 start)")
axes[0].set_title("Holdout equity curves — top-5 strategies vs Buy & Hold (net of costs)",
                  fontweight="bold")
axes[0].legend(fontsize=7.5)

# drawdowns
for name, bt_df in named.items():
    dd = bt_df["equity"] / bt_df["equity"].cummax() - 1
    lw = 2.0 if "BUY" in name else 1.1
    axes[1].plot(bt_df.index, dd * 100, label=name, lw=lw)
axes[1].axhline(0, color="k", lw=0.7)
axes[1].set_ylabel("drawdown (%)"); axes[1].set_xlabel("date")
axes[1].set_title("Drawdown comparison")
axes[1].legend(fontsize=7.5)
plt.tight_layout(); plt.show()
""")

code(r"""
# ── Holdout leaderboard: EXCESS-return & total-vs-B&H comparison ──────────────
plot_lb = lb[lb.type != "benchmark"].head(12).copy()
plot_lb["label"] = plot_lb["strategy"].str[:28] + " [" + plot_lb["mode"].str[:2] + "]"
bh_ret = float(bh_row["total_return"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# LEFT: excess return vs B&H (the headline metric)
ex = plot_lb.sort_values("excess_return")
colors = ["seagreen" if e > 0 else "firebrick" for e in ex["excess_return"]]
axes[0].barh(range(len(ex)), ex["excess_return"] * 100, color=colors, edgecolor="none", alpha=0.85)
axes[0].axvline(0, color="steelblue", lw=2, ls="--", label="Buy & Hold (0 = tie)")
axes[0].set_yticks(range(len(ex))); axes[0].set_yticklabels(ex["label"], fontsize=7.5)
axes[0].set_xlabel("EXCESS return vs B&H (percentage points)"); axes[0].legend()
axes[0].set_title("Holdout EXCESS return over Buy & Hold\n(green = beats B&H)",
                  fontweight="bold")

# RIGHT: absolute total return with B&H reference line
tot = plot_lb.sort_values("total_return")
tcolors = ["seagreen" if t > bh_ret else "firebrick" for t in tot["total_return"]]
axes[1].barh(range(len(tot)), tot["total_return"] * 100, color=tcolors, edgecolor="none", alpha=0.85)
axes[1].axvline(bh_ret * 100, color="steelblue", lw=2, ls="--", label=f"B&H = {bh_ret:.1%}")
axes[1].set_yticks(range(len(tot))); axes[1].set_yticklabels(tot["label"], fontsize=7.5)
axes[1].set_xlabel("holdout total net return (%)"); axes[1].legend()
axes[1].set_title("Holdout total return vs Buy & Hold line", fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ======================================================= 11. WINNERS DEEP-DIVE
md("## 11. Winners per mode — deep-dive")

code(r"""
def best_of(mode):
    sub = lb[(lb["mode"] == mode) & (lb["type"] != "benchmark")]
    return sub.iloc[0] if len(sub) else None

for mode in ("long_only", "long_short"):
    w = best_of(mode)
    if w is None: continue
    beat = "BEATS  ✓" if w["excess_return"] > 0 else "TRAILS ✗"
    print(f"[{mode}] winner: {w['strategy']}")
    print(f"  return {w['total_return']:.1%}  vs  B&H {bh_row['total_return']:.1%}   "
          f"→  EXCESS {w['excess_return']:+.1%}  ({w['vs_bh_x']:.2f}× B&H)  {beat}")
    print(f"  Sharpe {w['sharpe']:.2f}  maxDD {w['max_drawdown']:.1%}  "
          f"Calmar {w['calmar']:.2f}  wins {w['win_rate']:.1%}  trades {int(w['n_trades'])}\n")

# best overall
best_key  = ranked_keys[0][0]
best_sig  = ranked_keys[0][1]
funding   = CFG.short_funding_bps_per_bar if best_key.endswith("long_short") else 0.0
best_bt   = backtest(best_sig, hold_ret, CFG.cost_bps_per_side, funding)
""")

code(r"""
# ── Best strategy deep-dive: 4-panel chart ───────────────────────────────────
bh_bt = buy_and_hold(hold_ret, CFG.cost_bps_per_side)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
label = best_key.replace("__", " | ")

# equity (log)
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(best_bt.index, best_bt["equity"], lw=2, color="seagreen", label=label)
ax0.plot(bh_bt.index,   bh_bt["equity"],   lw=2, color="steelblue", ls="--", label="BUY & HOLD")
ax0.set_yscale("log"); ax0.legend(); ax0.set_ylabel("equity (log)")
ax0.set_title(f"Best strategy vs Buy & Hold — holdout (net of costs)", fontweight="bold")

# drawdown
ax1 = fig.add_subplot(gs[1, 0])
for name_p, bt_p, color in [(label, best_bt,"seagreen"),("B&H", bh_bt,"steelblue")]:
    dd = bt_p["equity"] / bt_p["equity"].cummax() - 1
    ax1.fill_between(bt_p.index, dd * 100, 0, alpha=0.45, color=color, label=name_p)
ax1.set_title("Drawdown (%)"); ax1.set_ylabel("%"); ax1.legend(fontsize=8)

# rolling Sharpe
ax2 = fig.add_subplot(gs[1, 1])
rs = (best_bt["strat_ret"].rolling(45).mean() /
      best_bt["strat_ret"].rolling(45).std() * np.sqrt(CFG.periods_per_year))
ax2.plot(rs.index, rs, color="darkorange", lw=1.2)
ax2.axhline(0, color="k", lw=0.8); ax2.axhline(1, color="seagreen", ls="--", lw=0.9)
ax2.set_title("45-day rolling Sharpe"); ax2.set_ylabel("Sharpe")

# return histogram
ax3 = fig.add_subplot(gs[2, 0])
r_strat = best_bt["strat_ret"][best_bt["position"] != 0]
r_bh    = bh_bt["strat_ret"]
bins = np.linspace(np.percentile(r_bh, 1), np.percentile(r_bh, 99), 60)
ax3.hist(r_bh,    bins=bins, alpha=0.5, color="steelblue", label="B&H",  density=True)
ax3.hist(r_strat, bins=bins, alpha=0.6, color="seagreen",  label=label[:20], density=True)
ax3.axvline(0, color="k", lw=0.8); ax3.legend(fontsize=8)
ax3.set_title("Active-bar return distribution"); ax3.set_xlabel("daily return")

# position over time
ax4 = fig.add_subplot(gs[2, 1])
ax4.fill_between(best_bt.index, best_bt["position"], 0, alpha=0.55, color="seagreen",
                 where=best_bt["position"] > 0, label="long")
ax4.fill_between(best_bt.index, best_bt["position"], 0, alpha=0.55, color="firebrick",
                 where=best_bt["position"] < 0, label="short")
ax4.set_title("Position over holdout window"); ax4.set_ylabel("position")
ax4.set_ylim(-1.2, 1.2); ax4.legend(fontsize=8)

fig.suptitle(f"Best strategy deep-dive: {label}", fontsize=12, fontweight="bold")
plt.show()
""")

code(r"""
# ── Monthly P&L heatmap for best strategy ────────────────────────────────────
monthly_strat = (best_bt["strat_ret"] + 1).resample("ME").prod() - 1
monthly_bh    = (bh_bt["strat_ret"] + 1).resample("ME").prod() - 1

def make_pivot(monthly):
    p = monthly.to_frame("ret")
    p["year"] = p.index.year; p["month"] = p.index.month
    return p.pivot(index="year", columns="month", values="ret")

piv_s = make_pivot(monthly_strat)
piv_b = make_pivot(monthly_bh)
col_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(2, 1, figsize=(14, 6))
for ax, piv, title in [(axes[0], piv_s, f"Monthly P&L — {label[:35]}"),
                        (axes[1], piv_b, "Monthly P&L — Buy & Hold")]:
    piv.columns = col_names[:len(piv.columns)]
    lim = max(abs(np.nanpercentile(piv.values, 5)), abs(np.nanpercentile(piv.values, 95)))
    im = ax.imshow(piv.values, cmap=CMAP_DIV, aspect="auto", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    for i in range(len(piv)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < lim * 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.02, label="monthly return")
    ax.set_title(title, fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ======================================================= 12. ROBUSTNESS
md(r"""
## 12. Robustness & overfitting checks

A strategy that survives these four checks is far more likely to generalise:
cost sensitivity, circular-shift null, block-bootstrap CIs, and IS vs OOS gap.
""")

code(r"""
best_sig_hold = ranked_keys[0][1].reindex(hold_ret.index)

# 1) cost sensitivity
sweep_d = {best_key.split("__")[0]: cost_sweep(
    best_sig_hold, hold_ret, CFG.periods_per_year,
    cost_grid=(0, 5, 10, 20, 40, 80), short_funding_bps=funding)}
plots.cost_sweep_plot(sweep_d,
    title="Cost sensitivity — net return vs per-side cost (bps)"); plt.show()
print(list(sweep_d.values())[0].to_string())
""")

code(r"""
# 2) circular-shift null
null = circular_shift_null(best_sig_hold, hold_ret, CFG.cost_bps_per_side, n=600,
                           short_funding_bps=funding)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(null["nulls"], bins=45, alpha=0.7, color="gray", label="null (shifted signals)")
ax.axvline(null["real_return"], color="seagreen", lw=2.5,
           label=f"real = {null['real_return']:.1%}")
ax.axvline(null["null_p95"], color="orange", lw=1.8, ls="--",
           label=f"null p95 = {null['null_p95']:.1%}")
ax.set_title(f"Circular-shift null test  (p-value = {null['p_value']:.3f})",
             fontweight="bold")
ax.legend(); ax.set_xlabel("holdout net return"); plt.tight_layout(); plt.show()
print(f"p={null['p_value']:.3f}  {'✓ edge is real (p<0.05)' if null['p_value']<0.05 else '✗ edge not significant'}")
""")

code(r"""
# 3) block-bootstrap CI
ci = block_bootstrap_ci(best_bt["strat_ret"], block=15, n=2000,
                        ppy=CFG.periods_per_year)
print(f"Holdout total-return 90% CI : [{ci['total_return_ci'][0]:.1%}, {ci['total_return_ci'][1]:.1%}]")
print(f"Holdout Sharpe       90% CI : [{ci['sharpe_ci'][0]:.2f}, {ci['sharpe_ci'][1]:.2f}]")
print(f"P(holdout return < 0)       : {ci['p_return_negative']:.1%}")

# bootstrap histogram
from src.robustness import block_bootstrap_ci as _bsci
rng = np.random.default_rng(99)
r = best_bt["strat_ret"].to_numpy()
T = len(r); block = 15; n_blocks = int(np.ceil(T / block)); n_boot = 2000
tot_boot = []
for _ in range(n_boot):
    starts = rng.integers(0, T - block, n_blocks)
    sample = np.concatenate([r[s:s+block] for s in starts])[:T]
    tot_boot.append(np.prod(1 + sample) - 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(tot_boot, bins=60, alpha=0.75, color="steelblue")
ax.axvline(np.percentile(tot_boot, 5),  color="firebrick", lw=2, ls="--", label="5th pctl")
ax.axvline(np.percentile(tot_boot, 95), color="seagreen",  lw=2, ls="--", label="95th pctl")
ax.axvline(best_bt["equity"].iloc[-1]-1, color="k", lw=2, label=f"point est {best_bt['equity'].iloc[-1]-1:.1%}")
ax.axvline(0, color="gray", lw=1.0)
ax.set_title("Block-bootstrap distribution of holdout total return", fontweight="bold")
ax.legend(); ax.set_xlabel("total return"); plt.tight_layout(); plt.show()
""")

code(r"""
# 4) IS vs OOS gap — on EXCESS return vs B&H (dev folds vs holdout)
gap_rows = []
for _, row in lb[lb.type != "benchmark"].iterrows():
    gap_rows.append({"strategy": str(row["strategy"])[:25] + f" [{row['mode'][:2]}]",
                     "OOS_excess": row["excess_return"],
                     "dev_mean_excess": row.get("dev_mean_excess", np.nan)})
gap_df = pd.DataFrame(gap_rows).dropna(subset=["dev_mean_excess"]).head(12)
gap_df["IS_OOS_gap"] = gap_df["dev_mean_excess"] - gap_df["OOS_excess"]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(gap_df))
w = 0.35
ax.bar(x - w/2, gap_df["dev_mean_excess"] * 100, w, label="dev mean OOS excess vs B&H",
       color="steelblue", alpha=0.8)
ax.bar(x + w/2, gap_df["OOS_excess"] * 100, w, label="holdout excess vs B&H",
       color="seagreen", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(gap_df["strategy"], rotation=45, ha="right", fontsize=7.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("excess return vs B&H (%)")
ax.set_title("In-sample (dev folds) vs Out-of-sample (holdout) EXCESS return over B&H\n"
             "(smaller gap = less overfit; >0 = beats benchmark)", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print("IS→OOS gap on excess-vs-B&H (dev_mean − holdout):  larger = more overfit")
print(gap_df[["strategy","dev_mean_excess","OOS_excess","IS_OOS_gap"]].to_string(index=False))
""")

# ======================================================= 13. CONCLUSION
md(r"""
## 13. Conclusion

Key caveats to read alongside the verdict below:
- **Buy & hold of SOL since 2020 is an extraordinary benchmark (~215×).** The realistic
  win for a timing strategy is better **risk-adjusted return** (Sharpe / Calmar / smaller
  drawdown), which it can achieve by sitting out crashes.
- All numbers from the **once-touched holdout** — nothing was optimised against it.
- Fees + slippage included; the cost-sweep shows how much edge remains at higher costs.
- ~1.9k daily bars is small: prefer stable, simple configs and treat the null-test
  p-value and bootstrap CIs as the honest confidence check.
""")

code(r"""
print("="*72)
print("FINAL VERDICT  (holdout, net of costs)")
print("="*72)
print(f"Buy & Hold      : return {bh_row['total_return']:>8.1%} | excess  +0.0% | "
      f"Sharpe {bh_row['sharpe']:>5.2f} | maxDD {bh_row['max_drawdown']:>7.1%}")
for mode in ("long_only", "long_short"):
    w = best_of(mode)
    if w is None: continue
    beat = "BEATS  ✓" if w["excess_return"] > 0 else "TRAILS ✗"
    print(f"Best {mode:<12}: return {w['total_return']:>8.1%} | excess {w['excess_return']:>+6.1%} | "
          f"Sharpe {w['sharpe']:>5.2f} | maxDD {w['max_drawdown']:>7.1%}  "
          f"{beat}  <- {w['strategy']}")
ov = lb.iloc[0]   # leaderboard is sorted by excess_return
print("-"*72)
print(f"Biggest alpha over B&H  : {ov['strategy']} [{ov['mode']}] = "
      f"{ov['excess_return']:+.1%} excess  ({ov['vs_bh_x']:.2f}× B&H, "
      f"total {ov['total_return']:.1%}, Sharpe {ov['sharpe']:.2f})")
best_shp = lb[lb.type != "benchmark"].sort_values("sharpe", ascending=False).iloc[0]
print(f"Best risk-adjusted      : {best_shp['strategy']} [{best_shp['mode']}] "
      f"= Sharpe {best_shp['sharpe']:.2f} (excess {best_shp['excess_return']:+.1%})")
print("="*72)
print(f"\nNull-test p-value       : {null['p_value']:.3f}  "
      f"{'(edge likely real)' if null['p_value'] < 0.05 else '(edge NOT significant at 5%)'}")
print(f"Bootstrap return 90% CI : [{ci['total_return_ci'][0]:.1%}, {ci['total_return_ci'][1]:.1%}]")
""")

# ================================================================ build
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_sol_strategy_research.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"wrote notebooks/01_sol_strategy_research.ipynb with {len(cells)} cells")

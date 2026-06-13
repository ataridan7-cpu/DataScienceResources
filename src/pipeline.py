"""High-level pipeline stages shared by the notebook and the runner script.

Heavy stages cache their outputs under results/, so re-running the notebook
is fast and deterministic while remaining fully reproducible (delete the
cache or pass force=True to recompute).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest, buy_and_hold, proba_to_signal
from .config import CFG, Config
from .cv import PurgedWalkForward, dev_holdout_split, holdout_windows
from .data_io import load_prices, load_btc_data
from .features import build_features
from .labels import forward_return, sign_label
from .metrics import perf_metrics
from .models import RULES, make_model
from .selection import build_feature_sets
from .tuning import (SEQ_ARCHS, eval_signal_on_folds, grid_search_rules,
                     ml_eval_config, run_study, split_params)

HORIZONS = (1, 2, 3, 5, 10, 21)
MAX_HORIZON = max(HORIZONS)


# ------------------------------------------------------------------ dataset
def load_dataset(cfg: Config = CFG) -> dict:
    """Load, clean, feature-engineer and split. Returns everything downstream needs."""
    cfg.ensure_dirs()
    df = load_prices(cfg.raw_csv)
    btc = load_btc_data(cfg.processed_dir / "btc.parquet")
    X_all, groups = build_features(df, btc=btc)
    valid = X_all.dropna().index            # drop indicator warmup rows
    X = X_all.loc[valid]
    prices = df.loc[valid]
    ret = df["close"].pct_change().loc[valid]
    fwd = {h: forward_return(df["close"], h).loc[valid] for h in HORIZONS}
    y1 = sign_label(df["close"], 1).loc[valid]

    n = len(X)
    dev_idx, hold_idx = dev_holdout_split(n, cfg.holdout_frac)
    folds = list(PurgedWalkForward(cfg.n_folds, cfg.min_train_frac,
                                   purge=MAX_HORIZON,
                                   embargo=cfg.embargo_bars,
                                   sliding=cfg.cv_sliding).split(len(dev_idx)))
    return {
        "df": df, "X": X, "prices": prices, "ret": ret, "fwd": fwd, "y1": y1,
        "groups": groups, "dev_idx": dev_idx, "hold_idx": hold_idx, "folds": folds,
        "dev_slice": slice(0, len(dev_idx)), "hold_slice": slice(len(dev_idx), n),
    }


def _dev(data, obj):
    return obj.iloc[data["dev_idx"]]


# ------------------------------------------------------------------- stages
def stage_rules(data, cfg: Config = CFG, force: bool = False) -> dict[str, pd.DataFrame]:
    """Exhaustive rule grids on dev folds for both modes (cached)."""
    out = {}
    for mode in ("long_only", "long_short"):
        cache = cfg.results_dir / "tuning" / f"rules__{mode}.csv"
        if cache.exists() and not force:
            out[mode] = pd.read_csv(cache)
            continue
        lb = grid_search_rules(_dev(data, data["prices"]), _dev(data, data["ret"]),
                               data["folds"], mode, cfg.cost_bps_per_side,
                               cfg.short_funding_bps_per_bar, cfg.periods_per_year)
        lb.to_csv(cache, index=False)
        out[mode] = lb
    return out


def stage_feature_sets(data, cfg: Config = CFG, force: bool = False) -> dict:
    return build_feature_sets(_dev(data, data["X"]), _dev(data, data["y1"]),
                              _dev(data, data["ret"]), data["folds"],
                              cfg.results_dir / "tuning" / "feature_sets.json",
                              cfg.cost_bps_per_side, cfg.seed, force, n_jobs=cfg.n_jobs)


def stage_ml(data, feature_sets: dict, models: list[str], cfg: Config = CFG,
             force: bool = False) -> dict:
    """Optuna studies for each model family x mode (cached per study)."""
    results = {}
    for mode in ("long_only", "long_short"):
        for name in models:
            key = f"{name}__{mode}"
            results[key] = run_study(
                name, mode, _dev(data, data["X"]),
                {h: _dev(data, s) for h, s in data["fwd"].items()},
                _dev(data, data["ret"]), data["folds"], feature_sets["sets"],
                cfg.cost_bps_per_side, cfg.short_funding_bps_per_bar,
                cfg.results_dir / "tuning", seed=cfg.seed,
                ppy=cfg.periods_per_year, force=force, n_jobs=cfg.n_jobs)
    return results


# --------------------------------------------------------------- final eval
def holdout_signal_ml(data, model_name: str, flat_params: dict, mode: str,
                      feature_sets: dict, cfg: Config = CFG) -> pd.Series:
    """Refit best config on the FULL dev set, emit causal signal over the holdout."""
    strat, mparams = split_params(flat_params)
    cols = feature_sets["sets"][strat["feature_set"]]
    h = strat["horizon"]
    band = 0.0 if model_name in SEQ_ARCHS else strat.get("band", 0.0)

    X, fwd = data["X"], data["fwd"][h]
    dev, hold = data["dev_idx"], data["hold_idx"]
    fr_dev = fwd.iloc[dev]
    if model_name in SEQ_ARCHS:
        Xtr = X.iloc[dev][cols]
        ytr = (fr_dev.fillna(0) > 0).astype(int).values
    else:
        mask = fr_dev.notna() & (fr_dev.abs() > band)
        Xtr = X.iloc[dev].loc[mask.values, cols]
        ytr = (fr_dev[mask] > 0).astype(int).values

    model = make_model(model_name, mparams, cfg.seed)
    model.fit(Xtr.values, ytr)
    p = pd.Series(model.predict_proba(X.iloc[hold][cols].values)[:, 1], index=X.index[hold])
    return proba_to_signal(p, mode=mode, sizing=strat["sizing"],
                           thr_long=strat.get("thr_long", 0.55),
                           thr_short=strat.get("thr_short", 0.45),
                           scale=strat.get("scale", 0.2))


def holdout_signal_rule(data, rule_name: str, params: dict, mode: str) -> pd.Series:
    """Rules need no fitting: compute over full history, slice the holdout."""
    sig = RULES[rule_name]["fn"](data["prices"], mode=mode, **params)
    return sig.iloc[data["hold_idx"]]


def final_leaderboard(data, ml_results: dict, rules_results: dict, feature_sets: dict,
                      cfg: Config = CFG, top_rules: int = 2) -> tuple[pd.DataFrame, dict]:
    """One-shot holdout evaluation of every tuned family + top rules + B&H.

    Every strategy row carries `excess_return` (= total_return − B&H, in return
    points — always meaningful), `profitable`/`beats_bh` flags, and `vs_bh_x`
    (growth-multiple vs B&H). The multiple is ONLY defined when both the
    strategy and B&H finished in profit; otherwise the ratio is misleading
    (it explodes when B&H < 0, and dresses up a money-loser as a "winner" when
    both are negative), so it is left NaN and the additive excess is used.
    """
    hold_ret = data["ret"].iloc[data["hold_idx"]]
    bh = perf_metrics(buy_and_hold(hold_ret, cfg.cost_bps_per_side), cfg.periods_per_year)
    bh_ret = bh["total_return"]

    def _vs_bh(m: dict) -> dict:
        tr = m["total_return"]
        excess = tr - bh_ret
        # growth-multiple is an honest "N× B&H" only when BOTH legs are profitable
        vs = (1 + tr) / (1 + bh_ret) if (tr > 0 and bh_ret > 0) else np.nan
        return {"excess_return": excess, "vs_bh_x": vs,
                "profitable": bool(tr > 0), "beats_bh": bool(excess > 0)}

    entries, signals = [], {}

    for key, res in ml_results.items():
        if not res.get("best_params"):
            continue
        name, mode = key.split("__")
        sig = holdout_signal_ml(data, name, res["best_params"], mode, feature_sets, cfg)
        funding = cfg.short_funding_bps_per_bar if mode == "long_short" else 0.0
        m = perf_metrics(backtest(sig, hold_ret, cfg.cost_bps_per_side, funding),
                         cfg.periods_per_year)
        attrs = res.get("best_attrs") or {}
        entries.append({"strategy": name, "type": "ml", "mode": mode, "key": key,
                        "dev_score": res["best_score"],
                        "dev_mean_excess": attrs.get("mean_excess_return"),
                        "dev_mean_return": attrs.get("mean_return"),
                        **m, **_vs_bh(m)})
        signals[key] = sig

    for mode, lb in rules_results.items():
        for _, row in lb.head(top_rules).iterrows():
            params = json.loads(row["params"])
            sig = holdout_signal_rule(data, row["rule"], params, mode)
            funding = cfg.short_funding_bps_per_bar if mode == "long_short" else 0.0
            m = perf_metrics(backtest(sig, hold_ret, cfg.cost_bps_per_side, funding),
                             cfg.periods_per_year)
            key = f"{row['rule']}({row['params']})__{mode}"
            entries.append({"strategy": f"{row['rule']} {row['params']}", "type": "rule",
                            "mode": mode, "key": key, "dev_score": row["score"],
                            "dev_mean_excess": row.get("mean_excess_return"),
                            "dev_mean_return": row["mean_return"], **m, **_vs_bh(m)})
            signals[key] = sig

    entries.append({"strategy": "BUY & HOLD", "type": "benchmark", "mode": "long_only",
                    "key": "BUY & HOLD", "dev_score": np.nan, "dev_mean_excess": np.nan,
                    "dev_mean_return": np.nan, **bh,
                    "excess_return": 0.0, "vs_bh_x": 1.0,
                    "profitable": bool(bh_ret > 0), "beats_bh": False})

    # Ranked by DEV score — the ONLY legitimate selection criterion (it never
    # touches the holdout). Holdout columns are the out-of-sample CONSEQUENCE of
    # that pick, never the basis for it: sorting by holdout excess and crowning
    # the max is selection-on-test (winner's curse) and is deliberately avoided.
    lb = (pd.DataFrame(entries)
          .sort_values("dev_score", ascending=False, na_position="last")
          .reset_index(drop=True))
    return lb, signals


def rolling_window_eval(
    data, ml_results: dict, rules_results: dict, feature_sets: dict,
    cfg: Config = CFG, n_windows: int = 3, top_rules: int = 2,
    keys_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate strategies across n_windows sub-windows of the holdout.

    ML models: frozen hyperparams, refit on all data before each window
    (expanding training set) → each window is a genuine OOS evaluation.
    Rules: no refitting; full-history signal sliced per window.

    keys_filter: if set, only evaluate the listed strategy keys.
    """
    hold_idx = data["hold_idx"]
    windows = holdout_windows(hold_idx, n_windows)
    rows = []

    def _bh_m(w_idx):
        return perf_metrics(buy_and_hold(data["ret"].iloc[w_idx], cfg.cost_bps_per_side),
                            cfg.periods_per_year)

    bh_per_win = [_bh_m(w) for w in windows]

    def _period(w_idx):
        s = data["X"].index[w_idx[0]].strftime("%Y-%m-%d")
        e = data["X"].index[w_idx[-1]].strftime("%Y-%m-%d")
        return f"{s}–{e}"

    # ---- ML models ----
    for key, res in ml_results.items():
        if not res.get("best_params"):
            continue
        if keys_filter is not None and key not in keys_filter:
            continue
        name, mode = key.split("__")
        strat, mparams = split_params(res["best_params"])
        cols = feature_sets["sets"][strat["feature_set"]]
        h = strat["horizon"]
        band = 0.0 if name in SEQ_ARCHS else strat.get("band", 0.0)
        fr = data["fwd"][h]
        funding = cfg.short_funding_bps_per_bar if mode == "long_short" else 0.0

        for w_i, w_idx in enumerate(windows):
            train_idx = np.arange(0, w_idx[0])
            fr_tr = fr.iloc[train_idx]
            if name in SEQ_ARCHS:
                valid = fr_tr.notna().values
                last = int(valid.nonzero()[0][-1]) + 1 if valid.any() else 0
                Xtr = data["X"].iloc[train_idx][cols].iloc[:last]
                ytr = (fr_tr.iloc[:last] > 0).astype(int).values
            else:
                mask = fr_tr.notna() & (fr_tr.abs() > band)
                if mask.sum() < 20:
                    continue
                Xtr = data["X"].iloc[train_idx].loc[mask.values, cols]
                ytr = (fr_tr[mask] > 0).astype(int).values
            if len(ytr) < 20 or len(np.unique(ytr)) < 2:
                continue
            model = make_model(name, mparams, cfg.seed)
            model.fit(Xtr.values, ytr)
            p = pd.Series(
                model.predict_proba(data["X"].iloc[w_idx][cols].values)[:, 1],
                index=data["X"].index[w_idx],
            )
            sig = proba_to_signal(p, mode=mode, sizing=strat["sizing"],
                                  thr_long=strat.get("thr_long", 0.55),
                                  thr_short=strat.get("thr_short", 0.45),
                                  scale=strat.get("scale", 0.2))
            ret_w = data["ret"].iloc[w_idx]
            m = perf_metrics(backtest(sig, ret_w, cfg.cost_bps_per_side, funding),
                             cfg.periods_per_year)
            bh_m = bh_per_win[w_i]
            rows.append({
                "key": key, "strategy": name, "mode": mode,
                "window": w_i + 1, "n_bars": len(w_idx),
                "period": _period(w_idx),
                "total_return": m["total_return"],
                "bh_return": bh_m["total_return"],
                "excess_return": m["total_return"] - bh_m["total_return"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "beats_bh": bool(m["total_return"] > bh_m["total_return"]),
            })

    # ---- Rules ----
    for mode, lb_df in rules_results.items():
        funding = cfg.short_funding_bps_per_bar if mode == "long_short" else 0.0
        for _, row in lb_df.head(top_rules).iterrows():
            params = json.loads(row["params"])
            key = f"{row['rule']}({row['params']})__{mode}"
            if keys_filter is not None and key not in keys_filter:
                continue
            sig_full = RULES[row["rule"]]["fn"](data["prices"], mode=mode, **params)
            for w_i, w_idx in enumerate(windows):
                sig_w = sig_full.iloc[w_idx]
                ret_w = data["ret"].iloc[w_idx]
                m = perf_metrics(backtest(sig_w, ret_w, cfg.cost_bps_per_side, funding),
                                 cfg.periods_per_year)
                bh_m = bh_per_win[w_i]
                rows.append({
                    "key": key,
                    "strategy": f"{row['rule']} {row['params']}",
                    "mode": mode,
                    "window": w_i + 1, "n_bars": len(w_idx),
                    "period": _period(w_idx),
                    "total_return": m["total_return"],
                    "bh_return": bh_m["total_return"],
                    "excess_return": m["total_return"] - bh_m["total_return"],
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "beats_bh": bool(m["total_return"] > bh_m["total_return"]),
                })

    # ---- B&H benchmark ----
    for w_i, w_idx in enumerate(windows):
        bh_m = bh_per_win[w_i]
        rows.append({
            "key": "BUY & HOLD", "strategy": "BUY & HOLD", "mode": "long_only",
            "window": w_i + 1, "n_bars": len(w_idx),
            "period": _period(w_idx),
            "total_return": bh_m["total_return"],
            "bh_return": bh_m["total_return"],
            "excess_return": 0.0,
            "sharpe": bh_m["sharpe"],
            "max_drawdown": bh_m["max_drawdown"],
            "beats_bh": False,
        })

    return pd.DataFrame(rows)

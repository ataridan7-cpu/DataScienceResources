"""Profit-first hyperparameter search.

Objective (maximized): mean fold (model_profit − buy_and_hold) across CV folds.
Pure excess-return over B&H — no variance penalty. B&H is computed per fold
window so every trial's score is directly comparable to the benchmark.
Every trial runs the full signal -> position -> cost-aware backtest pipeline.
Rules use exhaustive grid search; ML families use Optuna TPE + MedianPruner.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest, buy_and_hold, proba_to_signal
from .metrics import perf_metrics
from .models import RULES, make_model

SEQ_ARCHS = {"lstm", "transformer"}

# Equal trial budgets — all classical/tree families get the same search depth.
# Neural nets are capped lower due to wall-clock cost (~60s/trial on CPU).
TRIAL_BUDGET = {
    "logistic": 80, "knn": 80, "svm_rbf": 80, "random_forest": 80,
    "xgboost": 80, "lightgbm": 80, "catboost": 80,
    "mlp": 40, "lstm": 32, "transformer": 24,
}


# ------------------------------------------------------------------ scoring
def score_folds(fold_stats: list[dict]) -> float:
    """Dispatch to the active scoring objective (CFG.scoring).

    Each fold_stat dict must contain at least:
        excess_return, sharpe_excess, calmar_excess
    Falls back gracefully to excess_return for missing keys.
    """
    from .config import CFG
    scoring = getattr(CFG, "scoring", "excess_return")
    key = {
        "excess_return": "excess_return",
        "excess_sharpe": "sharpe_excess",
        "sharpe":        "sharpe",
        "calmar":        "calmar_excess",
    }.get(scoring, "excess_return")
    vals = [f.get(key, f.get("excess_return", 0.0)) for f in fold_stats]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]
    return float(np.mean(vals)) if vals else -np.inf


def bh_fold_metrics(ret: pd.Series, folds, cost_bps: float, ppy: int = 365) -> list[dict]:
    """Per-fold buy & hold metrics. B&H is always fully long with no funding, so
    this is identical across every trial and both modes — compute it ONCE and
    reuse, instead of re-backtesting it inside every Optuna trial."""
    return [perf_metrics(buy_and_hold(ret.iloc[va], cost_bps), ppy) for _, va in folds]


def _build_fold_stat(m: dict, bh_m: dict) -> dict:
    """Compute all per-fold metrics in one place."""
    excess = m["total_return"] - bh_m["total_return"]
    sharpe_excess = m["sharpe"] - bh_m["sharpe"]
    cm = m.get("calmar", np.nan)
    bh_cm = bh_m.get("calmar", np.nan)
    calmar_excess = (cm - bh_cm) if (np.isfinite(cm) and np.isfinite(bh_cm)) else 0.0
    return {
        "total_return":   m["total_return"],
        "bh_return":      bh_m["total_return"],
        "excess_return":  excess,
        "sharpe":         m["sharpe"],
        "sharpe_excess":  sharpe_excess,
        "calmar":         cm,
        "calmar_excess":  calmar_excess,
        "max_drawdown":   m["max_drawdown"],
        "n_trades":       m["n_trades"],
    }


def eval_signal_on_folds(signal: pd.Series, ret: pd.Series, folds, cost_bps: float,
                         short_funding_bps: float = 0.0, ppy: int = 365,
                         bh_folds: list[dict] | None = None) -> dict:
    """Backtest signal on each fold test window; score via CFG.scoring objective.

    `bh_folds` (optional): precomputed per-fold B&H metrics from bh_fold_metrics();
    avoids re-running the identical benchmark backtest on every call.
    """
    if bh_folds is None:
        bh_folds = bh_fold_metrics(ret, folds, cost_bps, ppy)
    fold_stats = []
    for (_, va), bh_m in zip(folds, bh_folds):
        m = perf_metrics(backtest(signal.iloc[va], ret.iloc[va], cost_bps, short_funding_bps), ppy)
        fold_stats.append(_build_fold_stat(m, bh_m))
    excesses = [f["excess_return"] for f in fold_stats]
    rets     = [f["total_return"]  for f in fold_stats]
    return {
        "score":                  score_folds(fold_stats),
        "mean_excess_return":     float(np.mean(excesses)),
        "mean_return":            float(np.mean(rets)),
        "std_return":             float(np.std(rets)),
        "mean_sharpe":            float(np.mean([f["sharpe"] for f in fold_stats])),
        "mean_sharpe_excess":     float(np.mean([f["sharpe_excess"] for f in fold_stats])),
        "pct_folds_beat_bh":      float(np.mean([e > 0 for e in excesses])),
        "pct_folds_profitable":   float(np.mean([r > 0 for r in rets])),
        "fold_stats":             fold_stats,
    }


# ------------------------------------------------------------- rules: grid
def grid_search_rules(df_prices: pd.DataFrame, ret: pd.Series, folds, mode: str,
                      cost_bps: float, short_funding_bps: float, ppy: int = 365) -> pd.DataFrame:
    """Exhaustive grid over every rule family; returns leaderboard sorted by score."""
    funding = short_funding_bps if mode == "long_short" else 0.0
    bh_folds = bh_fold_metrics(ret, folds, cost_bps, ppy)   # benchmark once, reuse for every combo
    rows = []
    for rname, spec in RULES.items():
        keys = list(spec["grid"].keys())
        for combo in itertools.product(*spec["grid"].values()):
            params = dict(zip(keys, combo))
            if spec["constraint"] and not spec["constraint"](params):
                continue
            signal = spec["fn"](df_prices, mode=mode, **params).reindex(ret.index)
            res = eval_signal_on_folds(signal, ret, folds, cost_bps, funding, ppy, bh_folds=bh_folds)
            rows.append({"rule": rname, "params": json.dumps(params), **{
                k: res[k] for k in ("score", "mean_excess_return", "mean_return",
                                    "std_return", "mean_sharpe", "pct_folds_beat_bh",
                                    "pct_folds_profitable")}})
    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))


# ------------------------------------------------------------ ML evaluation
def ml_eval_config(model_name: str, model_params: dict, strat: dict, X: pd.DataFrame,
                   fwd_returns: dict[int, pd.Series], ret: pd.Series, folds,
                   feature_sets: dict[str, list[str]], cost_bps: float,
                   short_funding_bps: float, mode: str, seed: int = 42,
                   ppy: int = 365, report_cb=None, bh_folds: list[dict] | None = None) -> dict:
    """Train per fold, map P(up) -> positions, backtest net of costs.

    `bh_folds` (optional): precomputed per-fold B&H metrics — see bh_fold_metrics().
    """
    h = strat["horizon"]
    band = 0.0 if model_name in SEQ_ARCHS else strat.get("band", 0.0)
    cols = feature_sets[strat["feature_set"]]
    fr = fwd_returns[h]
    funding = short_funding_bps if mode == "long_short" else 0.0
    if bh_folds is None:
        bh_folds = bh_fold_metrics(ret, folds, cost_bps, ppy)

    fold_stats, running = [], []
    for fold_i, (tr, va) in enumerate(folds):
        fr_tr = fr.iloc[tr]
        mask = fr_tr.notna() & (fr_tr.abs() > band)
        if mask.sum() < 50 or fr_tr[mask].gt(0).nunique() < 2:
            return {"score": -np.inf, "error": "degenerate training labels"}
        if model_name not in SEQ_ARCHS:
            Xtr = X.iloc[tr].loc[mask.values, cols]
            ytr = (fr_tr[mask] > 0).astype(int).values
        else:
            valid = fr_tr.notna().values
            last = int(valid.nonzero()[0][-1]) + 1 if valid.any() else 0
            Xtr = X.iloc[tr][cols].iloc[:last]
            ytr = (fr_tr.iloc[:last] > 0).astype(int).values

        model = make_model(model_name, model_params, seed)
        model.fit(Xtr.values, ytr)
        p_up = pd.Series(model.predict_proba(X.iloc[va][cols].values)[:, 1],
                         index=X.index[va])
        signal = proba_to_signal(p_up, mode=mode, sizing=strat["sizing"],
                                 thr_long=strat.get("thr_long", 0.55),
                                 thr_short=strat.get("thr_short", 0.45),
                                 scale=strat.get("scale", 0.2))
        m    = perf_metrics(backtest(signal, ret.iloc[va], cost_bps, funding), ppy)
        fs   = _build_fold_stat(m, bh_folds[fold_i])
        fold_stats.append(fs)
        running.append(fs["excess_return"])
        if report_cb is not None:
            report_cb(fold_i, float(np.mean(running)))

    excesses = [f["excess_return"] for f in fold_stats]
    rets     = [f["total_return"]  for f in fold_stats]
    return {
        "score":                score_folds(fold_stats),
        "mean_excess_return":   float(np.mean(excesses)),
        "mean_return":          float(np.mean(rets)),
        "std_return":           float(np.std(rets)),
        "mean_sharpe":          float(np.mean([f["sharpe"] for f in fold_stats])),
        "mean_sharpe_excess":   float(np.mean([f["sharpe_excess"] for f in fold_stats])),
        "pct_folds_beat_bh":    float(np.mean([e > 0 for e in excesses])),
        "pct_folds_profitable": float(np.mean([r > 0 for r in rets])),
        "fold_stats":           fold_stats,
    }


# ------------------------------------------------------- Optuna search space
def _suggest_strategy(trial, mode: str, model_name: str, set_names: list[str]) -> dict:
    strat = {
        "horizon":     trial.suggest_categorical("horizon", [1, 2, 3, 5, 10, 21]),
        "sizing":      trial.suggest_categorical("sizing", ["binary", "scaled", "kelly"]),
        "feature_set": trial.suggest_categorical("feature_set", set_names),
    }
    if model_name not in SEQ_ARCHS:
        strat["band"] = trial.suggest_float("band", 0.0, 0.01)
    if strat["sizing"] == "binary":
        strat["thr_long"] = trial.suggest_float("thr_long", 0.50, 0.65)
        if mode == "long_short":
            strat["thr_short"] = trial.suggest_float("thr_short", 0.35, 0.50)
    elif strat["sizing"] == "scaled":
        strat["scale"] = trial.suggest_float("scale", 0.05, 0.5)
    # kelly: no extra params — size is determined by 2p-1
    return strat


def _suggest_model(trial, name: str) -> dict:
    s = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    f, i, c = s
    if name == "logistic":
        return {"C": f("C", 1e-3, 10.0, log=True)}
    if name == "svm_rbf":
        return {"C": f("C", 0.1, 100.0, log=True), "gamma": f("gamma", 1e-4, 1.0, log=True)}
    if name == "knn":
        return {"n_neighbors": i("n_neighbors", 5, 100, log=True),
                "weights": c("weights", ["uniform", "distance"])}
    if name == "random_forest":
        return {"n_estimators": i("n_estimators", 100, 250), "max_depth": i("max_depth", 2, 8),
                "min_samples_leaf": i("min_samples_leaf", 5, 50),
                "max_features": f("max_features", 0.3, 1.0)}
    if name == "xgboost":
        return {"n_estimators": i("n_estimators", 50, 400), "max_depth": i("max_depth", 2, 6),
                "learning_rate": f("learning_rate", 0.01, 0.3, log=True),
                "subsample": f("subsample", 0.5, 1.0),
                "colsample_bytree": f("colsample_bytree", 0.5, 1.0),
                "min_child_weight": i("min_child_weight", 1, 20),
                "reg_lambda": f("reg_lambda", 0.1, 30.0, log=True)}
    if name == "lightgbm":
        return {"n_estimators": i("n_estimators", 50, 400),
                "num_leaves": i("num_leaves", 4, 64, log=True),
                "learning_rate": f("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": f("feature_fraction", 0.5, 1.0),
                "bagging_fraction": f("bagging_fraction", 0.5, 1.0), "bagging_freq": 1,
                "min_child_samples": i("min_child_samples", 5, 60),
                "reg_lambda": f("reg_lambda", 0.1, 30.0, log=True)}
    if name == "catboost":
        return {"iterations": i("iterations", 50, 400), "depth": i("depth", 2, 6),
                "learning_rate": f("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": f("l2_leaf_reg", 1.0, 30.0, log=True)}
    if name == "mlp":
        return {"hidden": c("hidden", [16, 32, 64]), "n_layers": i("n_layers", 1, 2),
                "dropout": f("dropout", 0.0, 0.5), "lr": f("lr", 1e-4, 1e-2, log=True),
                "weight_decay": f("weight_decay", 1e-6, 1e-2, log=True),
                "epochs": 150, "patience": 12}
    if name == "lstm":
        return {"hidden": c("hidden", [8, 16, 32]), "seq_len": c("seq_len", [8, 16, 24]),
                "dropout": f("dropout", 0.0, 0.5), "lr": f("lr", 1e-4, 1e-2, log=True),
                "weight_decay": f("weight_decay", 1e-6, 1e-2, log=True),
                "epochs": 120, "patience": 10}
    if name == "transformer":
        return {"hidden": c("hidden", [16, 32]), "n_heads": c("n_heads", [2, 4]),
                "seq_len": c("seq_len", [8, 16, 24]),
                "dropout": f("dropout", 0.0, 0.5), "lr": f("lr", 1e-4, 1e-2, log=True),
                "weight_decay": f("weight_decay", 1e-6, 1e-2, log=True),
                "epochs": 120, "patience": 10}
    raise ValueError(name)


_STRATEGY_KEYS = {"horizon", "band", "sizing", "thr_long", "thr_short", "scale", "feature_set"}


def split_params(flat: dict) -> tuple[dict, dict]:
    strat = {k: v for k, v in flat.items() if k in _STRATEGY_KEYS}
    model = {k: v for k, v in flat.items() if k not in _STRATEGY_KEYS}
    return strat, model


def run_study(model_name: str, mode: str, X, fwd_returns, ret, folds, feature_sets,
              cost_bps: float, short_funding_bps: float, cache_dir: Path,
              n_trials: int | None = None, seed: int = 42, ppy: int = 365,
              force: bool = False, n_jobs: int = 1) -> dict:
    """Run (or load cached) Optuna study for one model family x mode.

    `n_jobs` runs that many trials concurrently (threads). Models are pinned to
    one internal thread (see make_model) so the cores aren't oversubscribed.
    """
    import optuna

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{model_name}__{mode}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    n_trials = n_trials or TRIAL_BUDGET[model_name]
    set_names = list(feature_sets.keys())
    bh_folds = bh_fold_metrics(ret, folds, cost_bps, ppy)   # benchmark once for the whole study
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        strat = _suggest_strategy(trial, mode, model_name, set_names)
        mparams = _suggest_model(trial, model_name)

        def cb(step, running_mean):
            trial.report(running_mean, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = ml_eval_config(model_name, mparams, strat, X, fwd_returns, ret, folds,
                             feature_sets, cost_bps, short_funding_bps, mode,
                             seed=seed, ppy=ppy, report_cb=cb, bh_folds=bh_folds)
        trial.set_user_attr("mean_excess_return", res.get("mean_excess_return"))
        trial.set_user_attr("mean_return", res.get("mean_return"))
        trial.set_user_attr("mean_sharpe", res.get("mean_sharpe"))
        trial.set_user_attr("pct_folds_beat_bh", res.get("pct_folds_beat_bh"))
        trial.set_user_attr("pct_folds_profitable", res.get("pct_folds_profitable"))
        trial.set_user_attr("fold_stats", res.get("fold_stats"))
        return res["score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))

    done = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
    result = {
        "model": model_name, "mode": mode, "n_trials": len(study.trials),
        "best_score": float(study.best_value) if done else None,
        "best_params": study.best_params if done else None,
        "best_attrs": dict(study.best_trial.user_attrs) if done else None,
        "top5": [{"score": float(t.value), "params": t.params,
                  "attrs": dict(t.user_attrs)} for t in top],
    }
    cache.write_text(json.dumps(result, indent=2, default=str))
    return result

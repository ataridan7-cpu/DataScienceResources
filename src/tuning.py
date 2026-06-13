"""Profit-first hyperparameter search.

Objective (maximized): mean fold NET total return  -  0.25 * std(fold returns).
Every trial runs the full signal -> position -> cost-aware backtest pipeline on
each walk-forward validation fold; the final holdout never enters any study.
Rules use exhaustive grid search; ML families use Optuna TPE + MedianPruner.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest, proba_to_signal
from .metrics import perf_metrics
from .models import RULES, make_model

STABILITY_LAMBDA = 0.25
SEQ_ARCHS = {"lstm", "transformer"}

# Optuna trial budgets per family (profit-first: generous where cheap)
TRIAL_BUDGET = {
    "logistic": 80, "knn": 60, "svm_rbf": 40, "random_forest": 60,
    "xgboost": 120, "lightgbm": 120, "catboost": 60,
    "mlp": 20, "lstm": 18, "transformer": 14,
}


# ------------------------------------------------------------------ scoring
def score_folds(fold_returns: list[float]) -> float:
    fr = np.asarray(fold_returns, dtype=float)
    return float(fr.mean() - STABILITY_LAMBDA * fr.std())


def eval_signal_on_folds(signal: pd.Series, ret: pd.Series, folds, cost_bps: float,
                         short_funding_bps: float = 0.0, ppy: int = 365) -> dict:
    """Slice a (full-length, causal) signal into fold test windows and backtest each."""
    fold_stats = []
    for _, va in folds:
        bt = backtest(signal.iloc[va], ret.iloc[va], cost_bps, short_funding_bps)
        m = perf_metrics(bt, ppy)
        fold_stats.append({"total_return": m["total_return"], "sharpe": m["sharpe"],
                           "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"]})
    rets = [f["total_return"] for f in fold_stats]
    return {
        "score": score_folds(rets),
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "mean_sharpe": float(np.mean([f["sharpe"] for f in fold_stats])),
        "pct_folds_profitable": float(np.mean([r > 0 for r in rets])),
        "fold_stats": fold_stats,
    }


# ------------------------------------------------------------- rules: grid
def grid_search_rules(df_prices: pd.DataFrame, ret: pd.Series, folds, mode: str,
                      cost_bps: float, short_funding_bps: float, ppy: int = 365) -> pd.DataFrame:
    """Exhaustive grid over every rule family; returns leaderboard sorted by score."""
    funding = short_funding_bps if mode == "long_short" else 0.0
    rows = []
    for rname, spec in RULES.items():
        keys = list(spec["grid"].keys())
        for combo in itertools.product(*spec["grid"].values()):
            params = dict(zip(keys, combo))
            if spec["constraint"] and not spec["constraint"](params):
                continue
            signal = spec["fn"](df_prices, mode=mode, **params).reindex(ret.index)
            res = eval_signal_on_folds(signal, ret, folds, cost_bps, funding, ppy)
            rows.append({"rule": rname, "params": json.dumps(params), **{
                k: res[k] for k in ("score", "mean_return", "std_return",
                                    "mean_sharpe", "pct_folds_profitable")}})
    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))


# ------------------------------------------------------------ ML evaluation
def ml_eval_config(model_name: str, model_params: dict, strat: dict, X: pd.DataFrame,
                   fwd_returns: dict[int, pd.Series], ret: pd.Series, folds,
                   feature_sets: dict[str, list[str]], cost_bps: float,
                   short_funding_bps: float, mode: str, seed: int = 42,
                   ppy: int = 365, report_cb=None) -> dict:
    """Train per fold, map P(up) -> positions, backtest net of costs."""
    h = strat["horizon"]
    band = 0.0 if model_name in SEQ_ARCHS else strat.get("band", 0.0)
    cols = feature_sets[strat["feature_set"]]
    fr = fwd_returns[h]
    funding = short_funding_bps if mode == "long_short" else 0.0

    fold_stats, running = [], []
    for fold_i, (tr, va) in enumerate(folds):
        fr_tr = fr.iloc[tr]
        mask = fr_tr.notna() & (fr_tr.abs() > band)
        if mask.sum() < 50 or fr_tr[mask].gt(0).nunique() < 2:
            return {"score": -np.inf, "error": "degenerate training labels"}
        Xtr = X.iloc[tr].loc[mask.values, cols] if model_name not in SEQ_ARCHS \
            else X.iloc[tr][cols]          # seq nets need contiguous rows
        ytr = (fr_tr[mask] > 0).astype(int).values if model_name not in SEQ_ARCHS \
            else (fr_tr.fillna(0) > 0).astype(int).values

        model = make_model(model_name, model_params, seed)
        model.fit(Xtr.values, ytr)
        p_up = pd.Series(model.predict_proba(X.iloc[va][cols].values)[:, 1],
                         index=X.index[va])
        signal = proba_to_signal(p_up, mode=mode, sizing=strat["sizing"],
                                 thr_long=strat.get("thr_long", 0.55),
                                 thr_short=strat.get("thr_short", 0.45),
                                 scale=strat.get("scale", 0.2))
        bt = backtest(signal, ret.iloc[va], cost_bps, funding)
        m = perf_metrics(bt, ppy)
        fold_stats.append({"total_return": m["total_return"], "sharpe": m["sharpe"],
                           "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"]})
        running.append(m["total_return"])
        if report_cb is not None:
            report_cb(fold_i, float(np.mean(running)))

    rets = [f["total_return"] for f in fold_stats]
    return {
        "score": score_folds(rets),
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "mean_sharpe": float(np.mean([f["sharpe"] for f in fold_stats])),
        "pct_folds_profitable": float(np.mean([r > 0 for r in rets])),
        "fold_stats": fold_stats,
    }


# ------------------------------------------------------- Optuna search space
def _suggest_strategy(trial, mode: str, model_name: str, set_names: list[str]) -> dict:
    strat = {
        "horizon": trial.suggest_categorical("horizon", [1, 2, 3, 5]),
        "sizing": trial.suggest_categorical("sizing", ["binary", "scaled"]),
        "feature_set": trial.suggest_categorical("feature_set", set_names),
    }
    if model_name not in SEQ_ARCHS:
        strat["band"] = trial.suggest_float("band", 0.0, 0.01)
    if strat["sizing"] == "binary":
        strat["thr_long"] = trial.suggest_float("thr_long", 0.50, 0.65)
        if mode == "long_short":
            strat["thr_short"] = trial.suggest_float("thr_short", 0.35, 0.50)
    else:
        strat["scale"] = trial.suggest_float("scale", 0.05, 0.5)
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
        return {"n_estimators": i("n_estimators", 100, 400), "max_depth": i("max_depth", 2, 8),
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
              force: bool = False) -> dict:
    """Run (or load cached) Optuna study for one model family x mode."""
    import optuna

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{model_name}__{mode}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    n_trials = n_trials or TRIAL_BUDGET[model_name]
    set_names = list(feature_sets.keys())
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
                             seed=seed, ppy=ppy, report_cb=cb)
        trial.set_user_attr("mean_sharpe", res.get("mean_sharpe"))
        trial.set_user_attr("pct_folds_profitable", res.get("pct_folds_profitable"))
        trial.set_user_attr("mean_return", res.get("mean_return"))
        return res["score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

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

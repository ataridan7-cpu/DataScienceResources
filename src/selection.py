"""Feature selection / best-combination search.

All selection happens inside the walk-forward TRAIN/VALIDATION folds of the
development set only — the final holdout never influences feature choices.
Produces named candidate feature sets that the Optuna studies search over.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest, buy_and_hold, proba_to_signal
from .metrics import perf_metrics
from .tuning import score_folds, bh_fold_metrics

BASELINE_LGBM = dict(n_estimators=200, num_leaves=15, learning_rate=0.05,
                     min_child_samples=20, verbosity=-1, class_weight="balanced")


def corr_prune(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Greedy drop of one feature from any |corr| > threshold pair (keep-first)."""
    corr = X.corr().abs()
    keep: list[str] = []
    for col in X.columns:
        if all(corr.loc[col, k] <= threshold for k in keep):
            keep.append(col)
    return keep


def _fit_baseline(Xtr: pd.DataFrame, ytr: np.ndarray, seed: int):
    # single-threaded: greedy candidate scoring is parallelised across cores instead
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(random_state=seed, n_jobs=1, **BASELINE_LGBM)
    model.fit(Xtr.values, ytr)
    return model


def permutation_ranking(X: pd.DataFrame, y: pd.Series, folds, cols: list[str],
                        seed: int = 42, n_repeats: int = 5, n_jobs: int = 4) -> pd.Series:
    """Mean permutation importance (ROC-AUC on validation folds), aggregated."""
    from sklearn.inspection import permutation_importance

    agg = pd.Series(0.0, index=cols)
    n_used = 0
    for tr, va in folds:
        ytr, yva = y.iloc[tr], y.iloc[va]
        m_tr, m_va = ytr.notna(), yva.notna()
        if m_tr.sum() < 100 or yva[m_va].nunique() < 2:
            continue
        model = _fit_baseline(X.iloc[tr].loc[m_tr.values, cols], ytr[m_tr].astype(int).values, seed)
        r = permutation_importance(model, X.iloc[va].loc[m_va.values, cols].values,
                                   yva[m_va].astype(int).values, scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)
        agg += pd.Series(r.importances_mean, index=cols)
        n_used += 1
    return (agg / max(n_used, 1)).sort_values(ascending=False)


def _profit_of_set(cols: list[str], X, y, ret, folds, cost_bps, seed,
                   bh_folds: list[dict] | None = None) -> float:
    """Score a feature subset using CFG.scoring objective via baseline LGBM.

    `bh_folds`: precomputed per-fold B&H metrics (see tuning.bh_fold_metrics);
    avoids re-benchmarking on every candidate subset.
    """
    if bh_folds is None:
        bh_folds = bh_fold_metrics(ret, folds, cost_bps, 365)
    fold_stats = []
    for (tr, va), bh_m in zip(folds, bh_folds):
        ytr = y.iloc[tr]
        m = ytr.notna()
        model = _fit_baseline(X.iloc[tr].loc[m.values, cols], ytr[m].astype(int).values, seed)
        p = pd.Series(model.predict_proba(X.iloc[va][cols].values)[:, 1], index=X.index[va])
        sig = proba_to_signal(p, mode="long_only", sizing="binary", thr_long=0.55)
        m_strat = perf_metrics(backtest(sig, ret.iloc[va], cost_bps), 365)
        fold_stats.append({
            "excess_return":  m_strat["total_return"] - bh_m["total_return"],
            "sharpe_excess":  m_strat["sharpe"] - bh_m["sharpe"],
            "calmar_excess":  (m_strat.get("calmar", 0.0) or 0.0) - (bh_m.get("calmar", 0.0) or 0.0),
            "sharpe":         m_strat["sharpe"],
        })
    return score_folds(fold_stats)


def greedy_forward(X, y, ret, folds, ranked: list[str], cost_bps: float,
                   max_features: int = 15, min_features: int = 5, pool: int = 25,
                   seed: int = 42, n_jobs: int = 4) -> list[str]:
    """Greedy forward selection by OOS net profit from the top-`pool` ranked features.

    Adds the best-scoring candidate at each step; keeps going past the plateau
    until `min_features` are chosen so the set is usable, then stops when no
    further addition improves OOS profit (up to `max_features`).
    """
    from joblib import Parallel, delayed
    bh_folds = bh_fold_metrics(ret, folds, cost_bps, 365)   # benchmark once, reuse every candidate
    candidates = ranked[:pool]
    chosen: list[str] = []
    best = -np.inf
    while len(chosen) < max_features:
        pending = [c for c in candidates if c not in chosen]
        if not pending:
            break
        scores = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_profit_of_set)(chosen + [c], X, y, ret, folds, cost_bps, seed, bh_folds)
            for c in pending)
        cand, sc = max(zip(pending, scores), key=lambda kv: kv[1])
        if sc <= best + 1e-6 and len(chosen) >= min_features:
            break
        chosen.append(cand)
        best = max(best, sc)
    return chosen if chosen else ranked[:min_features]


def build_feature_sets(X: pd.DataFrame, y: pd.Series, ret: pd.Series, folds,
                       cache_path: Path, cost_bps: float, seed: int = 42,
                       force: bool = False, n_jobs: int = 4) -> dict:
    """Produce the named candidate sets searched by every Optuna study."""
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    pruned = corr_prune(X, 0.95)
    ranking = permutation_ranking(X, y, folds, pruned, seed, n_jobs=n_jobs)
    ranked = list(ranking.index)
    greedy = greedy_forward(X, y, ret, folds, ranked, cost_bps, seed=seed, n_jobs=n_jobs)

    sets = {
        "all_pruned": pruned,
        "top10": ranked[:10],
        "top15": ranked[:15],
        "top25": ranked[:25],
        "top30": ranked[:30],
        "greedy": greedy,
    }
    out = {"sets": sets, "ranking": {k: float(v) for k, v in ranking.items()},
           "n_original": int(X.shape[1]), "n_pruned": len(pruned)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out

"""Heavy-lifting runner: executes every search stage and caches results so the
notebook renders fast and deterministically.

    python run_pipeline.py            # uses caches where present
    python run_pipeline.py --force    # recompute everything from scratch

Outputs under results/:
    tuning/rules__{mode}.csv
    tuning/feature_sets.json
    tuning/{model}__{mode}.json        (one per Optuna study)
    summary.json                        (holdout leaderboard + winners)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CFG
from src.models import ML_MODELS
from src.pipeline import (load_dataset, stage_rules, stage_feature_sets, stage_ml,
                          final_leaderboard, rolling_window_eval)


def _json_safe(o):
    """Recursively coerce numpy scalars and non-finite floats (NaN/inf -> null)
    so summary.json is valid, strict-parseable JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def main(force: bool = False):
    t0 = time.time()
    print("[1/5] loading dataset + features ...", flush=True)
    data = load_dataset(CFG)
    print(f"      X={data['X'].shape}  dev={len(data['dev_idx'])}  "
          f"holdout={len(data['hold_idx'])}  folds={len(data['folds'])}", flush=True)

    print("[2/5] rule grids (both modes) ...", flush=True)
    t = time.time()
    rules = stage_rules(data, CFG, force=force)
    print(f"      done in {time.time()-t:.0f}s  "
          f"(long_only n={len(rules['long_only'])}, long_short n={len(rules['long_short'])})",
          flush=True)

    print("[3/5] feature selection / best combinations ...", flush=True)
    t = time.time()
    fsets = stage_feature_sets(data, CFG, force=force)
    print(f"      done in {time.time()-t:.0f}s  sets={ {k: len(v) for k, v in fsets['sets'].items()} }",
          flush=True)

    print(f"[4/5] Optuna studies: {len(ML_MODELS)} families x 2 modes ...", flush=True)
    for i, name in enumerate(ML_MODELS, 1):
        t = time.time()
        stage_ml(data, fsets, [name], CFG, force=force)
        print(f"      ({i}/{len(ML_MODELS)}) {name}: {time.time()-t:.0f}s", flush=True)
    ml_results = stage_ml(data, fsets, ML_MODELS, CFG, force=False)

    print("[5/5] holdout evaluation (winner pre-selected on DEV score) ...", flush=True)
    lb, _ = final_leaderboard(data, ml_results, rules, fsets, CFG)
    bh = lb[lb.strategy == "BUY & HOLD"].iloc[0]
    cand = lb[lb.type != "benchmark"]
    winner = cand.iloc[0]                       # lb is sorted by dev_score -> honest pick
    # winner's-curse diagnostic: does dev rank predict the holdout at all?
    dev = cand["dev_score"].to_numpy(dtype=float)
    oos = cand["excess_return"].to_numpy(dtype=float)
    rank_corr = float(pd.Series(dev).corr(pd.Series(oos), method="spearman"))
    holdout_max = cand.sort_values("excess_return", ascending=False).iloc[0]

    print("[+] rolling OOS stability (winner refitted on 3 expanding windows) ...", flush=True)
    t = time.time()
    rolling_df = rolling_window_eval(
        data, ml_results, rules, fsets, CFG,
        n_windows=3, keys_filter=[winner["key"]])
    print(f"    done in {time.time()-t:.0f}s", flush=True)
    winner_roll = rolling_df[rolling_df["key"] == winner["key"]]
    n_beats = int(winner_roll["beats_bh"].sum()) if len(winner_roll) > 0 else 0
    rolling_records = rolling_df.to_dict(orient="records")

    summary = {
        "leaderboard": lb.to_dict(orient="records"),
        "buy_hold_return": float(bh["total_return"]),
        "buy_hold_sharpe": float(bh["sharpe"]),
        "selection": "winner pre-committed by DEV score; holdout touched once as OOS check",
        "winner_by_dev": winner.to_dict(),
        "winner_excess_return": float(winner["excess_return"]),
        "dev_oos_spearman": rank_corr,
        "holdout_max_excess_diagnostic": holdout_max.to_dict(),
        "winner_rolling_windows": rolling_records,
        "winner_beats_bh_in_n_windows": n_beats,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (CFG.results_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False))

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"Buy & Hold holdout return: {bh['total_return']:.1%} (Sharpe {bh['sharpe']:.2f})")
    print(f"\nHONEST WINNER (highest DEV score, frozen before holdout):")
    print(f"  {winner['strategy']} [{winner['mode']}]  dev_score={winner['dev_score']:+.3f}")
    print(f"  -> holdout: return {winner['total_return']:+.1%}, excess vs B&H "
          f"{winner['excess_return']:+.1%}, Sharpe {winner['sharpe']:.2f}")
    print(f"\nWinner's-curse check: Spearman(dev_score, holdout_excess) = {rank_corr:+.3f}")
    print(f"  (~0 or negative => holdout rank is noise; the {holdout_max['strategy']} "
          f"[{holdout_max['mode']}] {holdout_max['excess_return']:+.1%} 'max' is luck, "
          f"NOT a fair pick)")
    if len(winner_roll) > 0:
        print(f"\nRolling stability — winner beats B&H in {n_beats}/{len(winner_roll)} windows:")
        for _, row in winner_roll.iterrows():
            print(f"  W{int(row['window'])} {row['period']}: "
                  f"return {row['total_return']:+.1%}, B&H {row['bh_return']:+.1%}, "
                  f"excess {row['excess_return']:+.1%}  "
                  f"{'✓' if row['beats_bh'] else '✗'}")
    print("\nDiagnostic — full holdout leaderboard (NOT the selection criterion):")
    disp = lb.copy()
    disp["outcome"] = np.where(disp["beats_bh"] & disp["profitable"], "beats B&H + profit",
                       np.where(disp["beats_bh"], "beats B&H but LOST $",
                       np.where(disp["profitable"], "profit but trails B&H", "lost $ + trails")))
    disp["vs_bh_x"] = disp["vs_bh_x"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "n/a")
    disp = disp.sort_values("excess_return", ascending=False)
    cols = ["strategy", "mode", "dev_score", "total_return", "excess_return", "vs_bh_x",
            "sharpe", "outcome"]
    print(disp[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)

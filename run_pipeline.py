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
                          final_leaderboard)


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

    print("[5/5] final one-shot holdout leaderboard ...", flush=True)
    lb, _ = final_leaderboard(data, ml_results, rules, fsets, CFG)
    bh = lb[lb.strategy == "BUY & HOLD"].iloc[0]
    best_strat = lb[lb.type != "benchmark"].iloc[0]   # sorted by excess_return
    summary = {
        "leaderboard": lb.to_dict(orient="records"),
        "buy_hold_return": float(bh["total_return"]),
        "buy_hold_sharpe": float(bh["sharpe"]),
        "best_by_excess": best_strat.to_dict(),
        "best_excess_return": float(best_strat["excess_return"]),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (CFG.results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"Buy & Hold holdout return: {bh['total_return']:.1%} (Sharpe {bh['sharpe']:.2f})")
    print("\nTop 8 by holdout EXCESS return vs Buy & Hold:")
    disp = lb.head(8).copy()
    # outcome makes the relative-vs-absolute distinction explicit so a strategy
    # that "beats B&H" while still losing money is never read as a winner
    disp["outcome"] = np.where(disp["beats_bh"] & disp["profitable"], "beats B&H + profit",
                       np.where(disp["beats_bh"], "beats B&H but LOST $",
                       np.where(disp["profitable"], "profit but trails B&H", "lost $ + trails")))
    # "× B&H" is undefined (NaN) unless both legs are profitable -> show as "n/a"
    disp["vs_bh_x"] = disp["vs_bh_x"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "n/a")
    cols = ["strategy", "mode", "total_return", "excess_return", "vs_bh_x",
            "sharpe", "max_drawdown", "n_trades", "outcome"]
    print(disp[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)

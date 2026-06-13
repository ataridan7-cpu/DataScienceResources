"""Central configuration for the SOL trading research pipeline."""
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # --- paths ---
    raw_csv: Path = REPO_ROOT / "data" / "raw" / "solana_2020-04-09_2025-06-14.csv"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    results_dir: Path = REPO_ROOT / "results"

    # --- transaction costs (basis points, charged per side on position change) ---
    fee_bps: float = 10.0        # taker fee
    slippage_bps: float = 10.0   # effective slippage
    short_funding_bps_per_bar: float = 1.0  # perp funding haircut while short (daily)

    # --- cross-validation ---
    n_folds: int = 6               # 6 folds → better regime coverage than 5
    min_train_frac: float = 0.40   # first walk-forward train window
    embargo_bars: int = 5          # extra gap between train end and test start
    holdout_frac: float = 0.15     # final never-touched test segment
    cv_sliding: bool = False       # True → fixed-size train window (non-expanding)

    # --- tuning objective ---
    # excess_return : mean(model_return − B&H) per fold  [old default]
    # excess_sharpe : mean(model_Sharpe − B&H Sharpe)    [rewards crash avoidance]
    # sharpe        : mean absolute Sharpe ratio
    # calmar        : mean(model_Calmar − B&H Calmar)
    scoring: str = "excess_sharpe"

    # --- misc ---
    seed: int = 42
    periods_per_year: int = 365    # crypto trades 7 days/week

    # --- parallelism ---
    n_jobs: int = 4                # parallel Optuna workers (= CPU cores)
    model_threads: int = 1         # internal threads per model fit

    @property
    def cost_bps_per_side(self) -> float:
        return self.fee_bps + self.slippage_bps

    def ensure_dirs(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "tuning").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures").mkdir(parents=True, exist_ok=True)


CFG = Config()

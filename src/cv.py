"""Purged walk-forward cross-validation for time series.

Expanding-window walk-forward: train always strictly precedes test.
`purge` drops the last bars of train whose forward-looking labels overlap the
test window; `embargo` adds an extra safety gap for serial correlation.
A final holdout (most recent `holdout_frac`) is carved off BEFORE any folds
and touched exactly once at the end of the research process.
"""
from __future__ import annotations

import numpy as np


class PurgedWalkForward:
    def __init__(self, n_folds: int = 5, min_train_frac: float = 0.40,
                 purge: int = 1, embargo: int = 5):
        self.n_folds = n_folds
        self.min_train_frac = min_train_frac
        self.purge = purge
        self.embargo = embargo

    def split(self, n: int):
        """Yield (train_idx, test_idx) positional index arrays over range(n)."""
        first_test = int(n * self.min_train_frac)
        bounds = np.linspace(first_test, n, self.n_folds + 1).astype(int)
        for k in range(self.n_folds):
            test_start, test_end = bounds[k], bounds[k + 1]
            train_end = max(0, test_start - self.purge - self.embargo)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def n_splits(self, n: int) -> int:
        return sum(1 for _ in self.split(n))


def dev_holdout_split(n: int, holdout_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Carve the most recent `holdout_frac` rows off as the final test segment."""
    cut = int(n * (1 - holdout_frac))
    return np.arange(0, cut), np.arange(cut, n)


def holdout_windows(hold_idx: np.ndarray, n_windows: int = 3) -> list[np.ndarray]:
    """Split holdout index into n_windows equal non-overlapping sub-windows."""
    return [arr for arr in np.array_split(hold_idx, n_windows) if len(arr) > 0]

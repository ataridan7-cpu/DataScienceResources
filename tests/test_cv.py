"""Purged walk-forward CV invariants: train precedes test with a purge+embargo gap."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import PurgedWalkForward, dev_holdout_split, holdout_windows  # noqa: E402


def test_train_strictly_precedes_test_with_gap():
    n, purge, embargo = 1000, 5, 5
    cv = PurgedWalkForward(n_folds=5, min_train_frac=0.40, purge=purge, embargo=embargo)
    folds = list(cv.split(n))
    assert len(folds) == 5
    for tr, va in folds:
        # no overlap, and train ends at least purge+embargo bars before test starts
        assert tr.max() < va.min()
        assert va.min() - tr.max() - 1 >= purge + embargo
        # test windows stay within range and are contiguous
        assert va.min() >= int(n * 0.40)
        assert np.array_equal(va, np.arange(va.min(), va.max() + 1))


def test_expanding_train_window():
    cv = PurgedWalkForward(n_folds=5, min_train_frac=0.40, purge=5, embargo=5)
    folds = list(cv.split(1000))
    train_ends = [tr.max() for tr, _ in folds]
    # expanding window: each fold trains on at least as much history as the last
    assert train_ends == sorted(train_ends)
    assert all(tr.min() == 0 for tr, _ in folds)  # always anchored at the start


def test_holdout_is_most_recent_and_disjoint():
    n = 1000
    dev, hold = dev_holdout_split(n, holdout_frac=0.15)
    assert hold.min() > dev.max()              # holdout is the most recent block
    assert len(hold) == n - int(n * 0.85)
    assert set(dev).isdisjoint(hold)
    assert len(dev) + len(hold) == n           # exhaustive partition


def test_holdout_windows_partition():
    _, hold = dev_holdout_split(1000, holdout_frac=0.15)
    windows = holdout_windows(hold, n_windows=3)
    assert len(windows) == 3
    # each window is non-empty and contiguous
    for w in windows:
        assert len(w) > 0
        assert np.array_equal(w, np.arange(w[0], w[-1] + 1))
    # windows are non-overlapping and cover the full holdout
    combined = np.concatenate(windows)
    assert set(combined) == set(hold)
    assert len(combined) == len(hold)
    # windows are ordered (earlier window precedes later)
    for i in range(len(windows) - 1):
        assert windows[i][-1] < windows[i + 1][0]


if __name__ == "__main__":
    for fn in [test_train_strictly_precedes_test_with_gap, test_expanding_train_window,
               test_holdout_is_most_recent_and_disjoint, test_holdout_windows_partition]:
        fn()
    print("cv tests passed")

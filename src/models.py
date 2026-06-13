"""Model zoo: rule-based technical strategies + ML classifiers + compact torch nets.

Uniform contracts
-----------------
- Rule strategies: ``fn(df, mode, **params) -> signal Series`` (decided at close t).
- ML models: ``make_model(name, params, seed)`` -> estimator with
  ``fit(X, y)`` and ``predict_proba(X)[:, 1]`` (P(up)).
Sequence nets receive the same tabular rows (time-ordered); they build windows
internally and emit a neutral 0.5 for the first ``seq_len - 1`` rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import sma, ema, rsi, macd, bollinger


# ====================================================================== rules
def _hysteresis(index, long_entry, long_exit, short_entry=None, short_exit=None) -> pd.Series:
    """Stateful entry/exit signal via event forward-fill (exits set first, entries win ties)."""
    ev = pd.Series(np.nan, index=index)
    ev[long_exit.fillna(False)] = 0.0
    if short_exit is not None:
        ev[short_exit.fillna(False)] = 0.0
    ev[long_entry.fillna(False)] = 1.0
    if short_entry is not None:
        ev[short_entry.fillna(False)] = -1.0
    return ev.ffill().fillna(0.0)


def rule_ma_cross(df: pd.DataFrame, mode: str = "long_only", fast: int = 20, slow: int = 100) -> pd.Series:
    diff = sma(df["close"], fast) - sma(df["close"], slow)
    sig = (diff > 0).astype(float)
    if mode == "long_short":
        sig = sig.where(diff.isna() | (diff > 0), -1.0)
    return sig.where(diff.notna(), 0.0)


def rule_macd(df: pd.DataFrame, mode: str = "long_only", fast: int = 12, slow: int = 26,
              signal: int = 9) -> pd.Series:
    _, _, hist = macd(df["close"], fast, slow, signal)
    sig = (hist > 0).astype(float)
    if mode == "long_short":
        sig = sig.where(hist > 0, -1.0)
    return sig


def rule_rsi(df: pd.DataFrame, mode: str = "long_only", n: int = 14, lo: float = 30,
             hi: float = 70, variant: str = "reversion") -> pd.Series:
    r = rsi(df["close"], n)
    if variant == "reversion":           # buy oversold, exit/short overbought
        long_entry, long_exit = r < lo, r > hi
        short_entry, short_exit = (r > hi, r < lo) if mode == "long_short" else (None, None)
    else:                                # momentum: buy strength
        long_entry, long_exit = r > hi, r < 50
        short_entry, short_exit = (r < lo, r > 50) if mode == "long_short" else (None, None)
    return _hysteresis(df.index, long_entry, long_exit, short_entry, short_exit)


def rule_bollinger(df: pd.DataFrame, mode: str = "long_only", n: int = 20, k: float = 2.0,
                   variant: str = "breakout") -> pd.Series:
    c = df["close"]
    _, _, upper, lower, mid = bollinger(c, n, k)
    if variant == "breakout":
        long_entry, long_exit = c > upper, c < mid
        short_entry, short_exit = (c < lower, c > mid) if mode == "long_short" else (None, None)
    else:                                # mean reversion
        long_entry, long_exit = c < lower, c > mid
        short_entry, short_exit = (c > upper, c < mid) if mode == "long_short" else (None, None)
    return _hysteresis(df.index, long_entry, long_exit, short_entry, short_exit)


def rule_donchian(df: pd.DataFrame, mode: str = "long_only", n_entry: int = 20,
                  n_exit: int = 10) -> pd.Series:
    c = df["close"]
    hi_prev = df["high"].rolling(n_entry).max().shift(1)
    lo_exit = df["low"].rolling(n_exit).min().shift(1)
    lo_prev = df["low"].rolling(n_entry).min().shift(1)
    hi_exit = df["high"].rolling(n_exit).max().shift(1)
    long_entry, long_exit = c > hi_prev, c < lo_exit
    short_entry, short_exit = (c < lo_prev, c > hi_exit) if mode == "long_short" else (None, None)
    return _hysteresis(df.index, long_entry, long_exit, short_entry, short_exit)


RULES: dict[str, dict] = {
    "ma_cross": {
        "fn": rule_ma_cross,
        "grid": {"fast": [5, 10, 15, 20, 30], "slow": [30, 50, 75, 100, 150, 200]},
        "constraint": lambda p: p["fast"] < p["slow"],
    },
    "macd": {
        "fn": rule_macd,
        "grid": {"fast": [8, 12, 16], "slow": [20, 26, 35], "signal": [6, 9, 12]},
        "constraint": lambda p: p["fast"] < p["slow"],
    },
    "rsi": {
        "fn": rule_rsi,
        "grid": {"n": [7, 14, 21], "lo": [20, 25, 30, 35], "hi": [65, 70, 75, 80],
                 "variant": ["reversion", "momentum"]},
        "constraint": None,
    },
    "bollinger": {
        "fn": rule_bollinger,
        "grid": {"n": [10, 20, 30], "k": [1.5, 2.0, 2.5], "variant": ["breakout", "reversion"]},
        "constraint": None,
    },
    "donchian": {
        "fn": rule_donchian,
        "grid": {"n_entry": [10, 20, 40, 55], "n_exit": [5, 10, 20]},
        "constraint": None,
    },
}


# ================================================================== ML models
ML_MODELS = ["logistic", "svm_rbf", "knn", "random_forest",
             "xgboost", "lightgbm", "catboost", "mlp", "lstm", "transformer"]


def make_model(name: str, params: dict, seed: int = 42, threads: int = 1):
    """Build an estimator pinned to `threads` internal threads.

    Trials are parallelised by Optuna (one thread per concurrent trial), so
    each model stays single-threaded to avoid oversubscribing the CPU.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=seed, **params))])
    if name == "svm_rbf":
        from sklearn.svm import SVC
        return Pipeline([("sc", StandardScaler()),
                         ("m", SVC(kernel="rbf", probability=True, class_weight="balanced",
                                   random_state=seed, **params))])
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsClassifier(n_jobs=threads, **params))])
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=threads,
                                      class_weight="balanced", **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=threads, eval_metric="logloss",
                             tree_method="hist", **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=threads, verbosity=-1,
                              class_weight="balanced", **params)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False,
                                  thread_count=threads, auto_class_weights="Balanced", **params)
    if name in ("mlp", "lstm", "transformer"):
        return TorchClassifier(arch=name, seed=seed, threads=threads, **params)
    raise ValueError(f"unknown model {name}")


# =========================================================== torch classifier
class TorchClassifier:
    """Compact CPU torch nets behind a sklearn-ish API.

    arch: 'mlp' (row-wise), 'lstm' or 'transformer' (sequence window over rows,
    which must be passed in time order). First seq_len-1 predictions are 0.5.
    """

    def __init__(self, arch: str = "mlp", hidden: int = 32, n_layers: int = 1,
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 200, patience: int = 15, batch_size: int = 64,
                 seq_len: int = 16, n_heads: int = 2, seed: int = 42, threads: int = 1):
        self.arch, self.hidden, self.n_layers = arch, hidden, n_layers
        self.dropout, self.lr, self.weight_decay = dropout, lr, weight_decay
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.seq_len, self.n_heads, self.seed = seq_len, n_heads, seed
        self.threads = threads

    # ------------------------------------------------------------- internals
    def _build(self, n_feat: int):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        h, p = self.hidden, self.dropout
        if self.arch == "mlp":
            layers, d = [], n_feat
            for _ in range(max(1, self.n_layers)):
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
                d = h
            layers += [nn.Linear(d, 1)]
            return nn.Sequential(*layers)
        if self.arch == "lstm":
            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.rnn = nn.LSTM(n_feat, h, num_layers=1, batch_first=True)
                    s.do = nn.Dropout(p)
                    s.fc = nn.Linear(h, 1)

                def forward(s, x):
                    out, _ = s.rnn(x)
                    return s.fc(s.do(out[:, -1]))
            return Net()
        # tiny transformer encoder
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj = nn.Linear(n_feat, h)
                layer = nn.TransformerEncoderLayer(d_model=h, nhead=self.n_heads,
                                                   dim_feedforward=2 * h, dropout=p,
                                                   batch_first=True)
                s.enc = nn.TransformerEncoder(layer, num_layers=1)
                s.fc = nn.Linear(h, 1)

            def forward(s, x):
                z = s.enc(s.proj(x))
                return s.fc(z[:, -1])
        return Net()

    def _sequences(self, Xs: np.ndarray) -> np.ndarray:
        L = self.seq_len
        if len(Xs) < L:
            return np.empty((0, L, Xs.shape[1]), dtype=np.float32)
        return np.stack([Xs[i - L + 1:i + 1] for i in range(L - 1, len(Xs))]).astype(np.float32)

    # ------------------------------------------------------------------- api
    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.set_num_threads(max(1, self.threads))  # avoid oversubscription under parallel trials
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = ((X - self.mu_) / self.sd_).astype(np.float32)

        if self.arch == "mlp":
            Xt, yt = Xs, y
        else:
            Xt = self._sequences(Xs)
            yt = y[self.seq_len - 1:]
            if len(Xt) == 0:
                raise ValueError("not enough rows for seq_len")

        # time-ordered early-stopping split (last 15%)
        cut = max(1, int(len(Xt) * 0.85))
        Xtr, ytr, Xva, yva = Xt[:cut], yt[:cut], Xt[cut:], yt[cut:]

        self.net_ = self._build(X.shape[1])
        pos_w = torch.tensor([max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1e-3)],
                             dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        Xtr_t = torch.tensor(Xtr)
        ytr_t = torch.tensor(ytr)
        Xva_t = torch.tensor(Xva) if len(Xva) else None
        yva_t = torch.tensor(yva) if len(Xva) else None

        best, best_state, bad = np.inf, None, 0
        n = len(Xtr_t)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.net_(Xtr_t[idx]).squeeze(-1)
                loss = loss_fn(out, ytr_t[idx])
                loss.backward()
                opt.step()
            if Xva_t is None or len(Xva_t) == 0:
                continue
            self.net_.eval()
            with torch.no_grad():
                vloss = loss_fn(self.net_(Xva_t).squeeze(-1), yva_t).item()
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float64)
        Xs = ((X - self.mu_) / self.sd_).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            if self.arch == "mlp":
                logits = self.net_(torch.tensor(Xs)).squeeze(-1).numpy()
                p = 1 / (1 + np.exp(-logits))
            else:
                seqs = self._sequences(Xs)
                p = np.full(len(Xs), 0.5)
                if len(seqs):
                    logits = self.net_(torch.tensor(seqs)).squeeze(-1).numpy()
                    p[self.seq_len - 1:] = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p, p])

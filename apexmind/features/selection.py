"""Statistical feature selection on training data only.

1. Drop features that are mostly missing or (near) constant.
2. Measure predictive stability: Spearman rank-IC against each target in
   contiguous time blocks; a feature's t-statistic is mean(IC)/se(IC) across
   blocks, which respects time-dependence better than a pooled correlation.
3. Remove redundancy: walk features in order of |t| and keep one only if its
   rank correlation with every kept feature is below a threshold.
4. Keep features whose best |t| over targets passes ``min_abs_t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    selected: list[str]
    t_stats: dict[str, float]
    dropped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"selected": self.selected, "t_stats": {k: round(v, 3) for k, v in self.t_stats.items()},
                "dropped": self.dropped}


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def _block_ic(x: np.ndarray, y: np.ndarray, n_blocks: int) -> np.ndarray:
    ics = []
    for xb, yb in zip(np.array_split(x, n_blocks), np.array_split(y, n_blocks)):
        m = ~(np.isnan(xb) | np.isnan(yb))
        if m.sum() < 30:
            continue
        xr, yr = _rank(xb[m]), _rank(yb[m])
        sx, sy = xr.std(), yr.std()
        if sx == 0 or sy == 0:
            continue
        ics.append(float(np.mean((xr - xr.mean()) * (yr - yr.mean())) / (sx * sy)))
    return np.asarray(ics)


def select_features(X: pd.DataFrame, targets: dict[str, np.ndarray], *, n_blocks: int = 8,
                    min_abs_t: float = 2.0, corr_threshold: float = 0.9, max_nan_frac: float = 0.5,
                    always_keep: tuple[str, ...] = ()) -> SelectionResult:
    dropped: dict[str, str] = {}
    cand = []
    for c in X.columns:
        col = X[c].to_numpy(dtype=float)
        nan_frac = np.isnan(col).mean()
        if nan_frac > max_nan_frac:
            dropped[c] = f"missing {nan_frac:.0%}"
            continue
        if np.nanstd(col) < 1e-12:
            dropped[c] = "constant"
            continue
        cand.append(c)
    tstats: dict[str, float] = {}
    for c in cand:
        x = X[c].to_numpy(dtype=float)
        best = 0.0
        for y in targets.values():
            ics = _block_ic(x, y, n_blocks)
            if len(ics) < 3:
                continue
            se = ics.std(ddof=1) / np.sqrt(len(ics))
            t = ics.mean() / se if se > 0 else 0.0
            if abs(t) > abs(best):
                best = float(t)
        tstats[c] = best
    order = sorted(cand, key=lambda c: -abs(tstats[c]))
    ranked = X[order].rank() if order else X[order]
    kept: list[str] = []
    for c in order:
        if abs(tstats[c]) < min_abs_t and c not in always_keep:
            dropped[c] = f"|t|={abs(tstats[c]):.2f} < {min_abs_t}"
            continue
        redundant = None
        for k in kept:
            rho = ranked[c].corr(ranked[k])
            if abs(rho) >= corr_threshold:
                redundant = k
                break
        if redundant is not None and c not in always_keep:
            dropped[c] = f"redundant with {redundant}"
            continue
        kept.append(c)
    return SelectionResult(kept, tstats, dropped)

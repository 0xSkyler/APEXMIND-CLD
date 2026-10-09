"""Chronological walk-forward folds with purge gaps.

Each fold::

    [fit ...........) gap [cal ......) gap [test ......)

The gap is at least the longest label span (latency + horizon + latency),
so no training or calibration label overlaps the next window. Training is
expanding (all history before the calibration window); test windows are
disjoint and later folds never influence earlier ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

H_NS = 3_600_000_000_000


@dataclass
class Fold:
    index: int
    fit_start: int
    fit_end: int
    cal_start: int
    cal_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in list(d):
            if k != "index":
                d[k + "_utc"] = pd.Timestamp(d[k], unit="ns", tz="UTC").isoformat()
        return d


def make_folds(t_min: int, t_max: int, n_folds: int, min_train_hours: float, test_hours: float,
               cal_fraction: float, gap_ns: int) -> list[Fold]:
    test_len = int(test_hours * H_NS)
    min_train = int(min_train_hours * H_NS)
    n = n_folds
    while n >= 1 and min_train + 2 * gap_ns + n * test_len > (t_max - t_min):
        n -= 1
    if n < 1:
        raise ValueError(
            f"data span {(t_max - t_min) / H_NS:.1f}h too short for min_train {min_train_hours}h + test {test_hours}h")
    folds = []
    for k in range(n):
        test_end = t_max - (n - 1 - k) * test_len
        test_start = test_end - test_len
        cal_end = test_start - gap_ns
        train_len = cal_end - t_min
        cal_start = cal_end - int(train_len * cal_fraction)
        fit_end = cal_start - gap_ns
        folds.append(Fold(k, t_min, fit_end, cal_start, cal_end, test_start, test_end))
    return folds


def window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    ts = df["ts_ns"].to_numpy()
    return df[(ts >= start) & (ts < end)]

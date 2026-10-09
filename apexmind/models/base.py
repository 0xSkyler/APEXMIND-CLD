"""Model interface.

A model maps point-in-time features to a *score* per (side, horizon); a
higher score means a better expected outcome for that side. Scores are not
trusted as return forecasts: the calibration step (:mod:`apexmind.research.policy`)
maps them to realized net returns on held-out data and learns thresholds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDES = ("long", "short")


@dataclass(frozen=True)
class TargetSpec:
    horizons: tuple[float, ...]

    def pairs(self) -> list[tuple[str, float]]:
        return [(s, h) for s in SIDES for h in self.horizons]

    @staticmethod
    def col(side: str, h: float) -> str:
        return f"ret_{side}_{h:g}"


class Model(ABC):
    name: str = "model"
    family: str = "baseline"
    complexity: int = 0  # simpler models win ties

    @abstractmethod
    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None: ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> dict[tuple[str, float], np.ndarray]: ...

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "complexity": self.complexity}


def block_ic_t(x: np.ndarray, y: np.ndarray, n_blocks: int = 8) -> float:
    """t-statistic of block-wise Spearman IC (time-robust)."""
    from apexmind.features.selection import _block_ic

    ics = _block_ic(x, y, n_blocks)
    if len(ics) < 3:
        return 0.0
    se = ics.std(ddof=1) / np.sqrt(len(ics))
    return float(ics.mean() / se) if se > 0 else 0.0


def training_rows(df: pd.DataFrame, spec: TargetSpec, max_rows: int, seed: int = 0) -> pd.DataFrame:
    """Valid rows with at least one label; evenly thinned to ``max_rows``.

    Thinning is systematic (every k-th row) rather than random: adjacent grid
    rows are near-duplicates and systematic thinning keeps time coverage.
    """
    cols = [TargetSpec.col(s, h) for s, h in spec.pairs()]
    m = df["valid"].to_numpy() & df[cols].notna().any(axis=1).to_numpy()
    sub = df[m]
    if len(sub) > max_rows:
        step = int(np.ceil(len(sub) / max_rows))
        sub = sub.iloc[::step]
    return sub

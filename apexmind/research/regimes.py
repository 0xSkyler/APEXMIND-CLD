"""Market-regime labels with thresholds learned on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeModel:
    vol_edges: tuple[float, float]
    spread_edge: float
    trend_edge: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "RegimeModel":
        v = train["ref_rv_bps"].dropna()
        s = train["lit_spread_bps"].dropna()
        tr = train["trend_60"].abs().dropna() if "trend_60" in train else pd.Series([1.0])
        return cls((float(v.quantile(1 / 3)), float(v.quantile(2 / 3))), float(s.median()), float(tr.quantile(0.8)))

    def label(self, rv: np.ndarray, spread: np.ndarray, trend: np.ndarray) -> pd.DataFrame:
        vol = np.where(rv <= self.vol_edges[0], "low_vol", np.where(rv <= self.vol_edges[1], "mid_vol", "high_vol"))
        vol = np.where(np.isnan(rv), "unknown", vol)
        spr = np.where(np.isnan(spread), "unknown", np.where(spread <= self.spread_edge, "tight_spread", "wide_spread"))
        trd = np.where(np.isnan(trend), "unknown", np.where(np.abs(trend) >= self.trend_edge, "trending", "ranging"))
        return pd.DataFrame({"vol_regime": vol, "spread_regime": spr, "trend_regime": trd})

    def to_dict(self) -> dict:
        return {"vol_edges_bps": self.vol_edges, "spread_edge_bps": self.spread_edge, "trend_edge": self.trend_edge}

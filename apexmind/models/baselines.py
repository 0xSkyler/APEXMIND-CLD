"""Simple, interpretable baselines (directive section 4, items 1-4).

Each picks its single free parameter (lookback) on the training data by
time-blocked IC; thresholds are learned later by the shared calibrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexmind.models.base import Model, TargetSpec, block_ic_t

FLOW_PREFIXES = ("lit_ofi_", "ref_ofi_", "lit_aggr_", "ref_aggr_")
FLOW_FIXED = ("lit_obi_l1", "ref_obi_l1", "lit_obi_depth", "ref_obi_depth")


def _lookbacks(df: pd.DataFrame, prefix: str, max_h: float = 15.0) -> list[str]:
    out = []
    for c in df.columns:
        if c.startswith(prefix):
            try:
                if float(c[len(prefix):]) <= max_h:
                    out.append(c)
            except ValueError:
                continue
    return out


class _SingleSignal(Model):
    prefix = ""

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.choice: dict[float, str] = {}
        self.ic: dict[float, float] = {}
        cands = _lookbacks(train, self.prefix)
        for h in spec.horizons:
            y = train[TargetSpec.col("long", h)].to_numpy(float)
            best, best_t = cands[0], -np.inf
            for c in cands:
                t = block_ic_t(train[c].to_numpy(float), y)
                if t > best_t:
                    best, best_t = c, t
            self.choice[h], self.ic[h] = best, float(best_t)

    def score(self, df: pd.DataFrame) -> dict:
        out = {}
        for h in self.spec.horizons:
            x = df[self.choice[h]].to_numpy(float)
            out[("long", h)] = x
            out[("short", h)] = -x
        return out

    def describe(self) -> dict:
        return {**super().describe(), "signal": {f"{h:g}": c for h, c in self.choice.items()},
                "train_ic_t": {f"{h:g}": round(t, 2) for h, t in self.ic.items()}}


class RefMomentum(_SingleSignal):
    """1. Reference-market momentum: trade Lighter in the direction of the
    reference venue's recent return."""

    name, complexity, prefix = "ref_momentum", 1, "ref_ret_"


class LagOnly(_SingleSignal):
    """2. Cross-venue lag: reference return minus Lighter return over the
    same window (the part of the reference move Lighter has not yet made)."""

    name, complexity, prefix = "lag_only", 1, "xret_"


class FlowOnly(Model):
    """3. Order-flow only: sign-weighted z-score composite of trade-flow, OFI
    and book-imbalance features with significant training IC."""

    name, complexity = "flow_only", 2

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        cols = [c for c in train.columns if c.startswith(FLOW_PREFIXES)] + [c for c in FLOW_FIXED if c in train]
        self.weights: dict[float, dict[str, tuple[float, float, float]]] = {}
        for h in spec.horizons:
            y = train[TargetSpec.col("long", h)].to_numpy(float)
            w = {}
            for c in cols:
                x = train[c].to_numpy(float)
                t = block_ic_t(x, y)
                if abs(t) >= 2.0:
                    w[c] = (float(np.sign(t)), float(np.nanmean(x)), float(np.nanstd(x)) or 1.0)
            self.weights[h] = w

    def composite(self, df: pd.DataFrame, h: float) -> np.ndarray:
        w = self.weights[h]
        if not w:
            return np.zeros(len(df))
        acc = np.zeros(len(df))
        for c, (sgn, mu, sd) in w.items():
            acc += sgn * np.nan_to_num((df[c].to_numpy(float) - mu) / sd)
        return acc / np.sqrt(len(w))

    def score(self, df: pd.DataFrame) -> dict:
        out = {}
        for h in self.spec.horizons:
            x = self.composite(df, h)
            out[("long", h)] = x
            out[("short", h)] = -x
        return out

    def describe(self) -> dict:
        return {**super().describe(), "features": {f"{h:g}": sorted(w) for h, w in self.weights.items()}}


class LagFlow(Model):
    """4. Cross-venue lag with order-flow confirmation: the lag signal only
    counts when the flow composite points the same way."""

    name, complexity = "lag_flow", 3

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.lag = LagOnly()
        self.lag.fit(train, spec)
        self.flow = FlowOnly()
        self.flow.fit(train, spec)

    def score(self, df: pd.DataFrame) -> dict:
        lag, out = self.lag.score(df), {}
        for h in self.spec.horizons:
            f = self.flow.composite(df, h)
            lo = lag[("long", h)].copy()
            lo[~(f > 0)] = -np.inf
            sh = lag[("short", h)].copy()
            sh[~(f < 0)] = -np.inf
            out[("long", h)], out[("short", h)] = lo, sh
        return out

    def describe(self) -> dict:
        return {**super().describe(), "lag": self.lag.describe(), "flow": self.flow.describe()}


class _Gated(Model):
    """Signal active only inside a regime defined by training quantiles."""

    prefix, gate_col, gate_q, sign = "", "", 0.8, 1.0

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.gate = float(np.nanquantile(train[self.gate_col].to_numpy(float), self.gate_q))
        self.choice = {}
        on = train[self.gate_col].to_numpy(float) > self.gate
        cands = _lookbacks(train, self.prefix, 5.0)
        for h in spec.horizons:
            y = train[TargetSpec.col("long", h)].to_numpy(float)[on]
            self.choice[h] = max(cands, key=lambda c: block_ic_t(self.sign * train[c].to_numpy(float)[on], y))

    def score(self, df: pd.DataFrame) -> dict:
        on = df[self.gate_col].to_numpy(float) > self.gate
        out = {}
        for h in self.spec.horizons:
            x = self.sign * df[self.choice[h]].to_numpy(float)
            out[("long", h)] = np.where(on, x, -np.inf)
            out[("short", h)] = np.where(on, -x, -np.inf)
        return out

    def describe(self) -> dict:
        return {**super().describe(), "gate": {self.gate_col: self.gate},
                "signal": {f"{h:g}": c for h, c in self.choice.items()}}


class ShockReversion(_Gated):
    """Secondary family: after a liquidity shock (spread far above its norm)
    fade the recent Lighter move."""

    name, complexity, prefix, gate_col, gate_q, sign = "shock_reversion", 2, "lit_ret_", "spread_ratio", 0.8, -1.0


class VolMomentum(_Gated):
    """Secondary family: follow reference momentum only in high-volatility
    regimes."""

    name, complexity, prefix, gate_col, gate_q, sign = "vol_momentum", 2, "ref_ret_", "vol_ratio", 0.67, 1.0

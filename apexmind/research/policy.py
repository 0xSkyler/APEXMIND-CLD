"""Calibrated, uncertainty-aware entry policy.

Fitted on a calibration window that the model never trained on:

* for each side and horizon, candidate thresholds are taken at upper
  quantiles of the model score;
* each candidate's realized *net executable* returns get a cluster-robust
  standard error (clusters = time blocks) and a lower confidence bound
  ``LCB = mean - z * se``;
* the (horizon, threshold) maximizing ``capacity * LCB`` is kept only if
  ``LCB > min_lcb``; otherwise that side never trades.

Long and short are calibrated separately with the same procedure. The
policy also records the empirical outcome distribution of the selected
calibration trades (std, quantiles, adverse excursion, passive-fill rates)
which the allocator and execution optimizer consume.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from apexmind.models.base import SIDES, TargetSpec
from apexmind.research.stats import cluster_mean_se, expected_shortfall, time_clusters

QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995)


@dataclass
class SidePolicy:
    side: str
    horizon: float
    threshold: float
    n_cal: int
    mean: float
    se: float
    lcb: float
    std: float
    p_pos: float
    q05: float
    q95: float
    es05: float
    mae_mean: float
    p_sig_slip: float
    hold_ms: float
    passive_fill: float = math.nan
    passive_mean_if_filled: float = math.nan
    passive_se: float = math.nan
    candidates: list = field(default_factory=list)
    _iso: IsotonicRegression | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_iso", None)
        return d

    def calibrated_mean(self, score: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return np.full(len(score), self.mean)
        return self._iso.predict(np.clip(score, -1e300, 1e300))


@dataclass
class Policy:
    model: str
    sides: dict[str, SidePolicy | None]
    rejected: dict[str, str]

    def to_dict(self) -> dict:
        return {"model": self.model, "rejected": self.rejected,
                "sides": {k: (v.to_dict() if v else None) for k, v in self.sides.items()}}

    def decide(self, scores: dict[tuple[str, float], np.ndarray], n: int) -> pd.DataFrame:
        """Per-row decision: side (+1/-1/0), horizon, calibrated mean, se."""
        side = np.zeros(n, dtype=np.int8)
        horizon = np.full(n, np.nan)
        mean = np.full(n, np.nan)
        se = np.full(n, np.nan)
        best = np.full(n, -np.inf)
        for name, sign in (("long", 1), ("short", -1)):
            sp = self.sides.get(name)
            if sp is None:
                continue
            s = scores[(name, sp.horizon)]
            mu = sp.calibrated_mean(s)
            mu = np.where(np.isnan(mu), sp.mean, mu)
            # if both sides fire, keep the larger calibrated expectation
            take = (s >= sp.threshold) & (mu > best)
            side[take] = sign
            horizon[take] = sp.horizon
            mean[take] = mu[take]
            se[take] = sp.se
            best[take] = mu[take]
        return pd.DataFrame({"side": side, "horizon": horizon, "mu": mean, "se": se})


def fit_policy(model_name: str, scores: dict, cal: pd.DataFrame, spec: TargetSpec, *, z: float, min_lcb: float,
               block_ns: int, min_trades: int, n_symbols: int, sig_slip_bps: float, passive: bool) -> Policy:
    ts = cal["ts_ns"].to_numpy()
    clusters = time_clusters(ts, block_ns)
    duration_s = max((ts.max() - ts.min()) / 1e9, 1.0) if len(ts) else 1.0
    sides: dict[str, SidePolicy | None] = {}
    rejected: dict[str, str] = {}
    for name in SIDES:
        best_obj, best = -np.inf, None
        cands = []
        for h in spec.horizons:
            r = cal[TargetSpec.col(name, h)].to_numpy(float)
            s = scores[(name, h)]
            ok = ~np.isnan(r) & np.isfinite(s)
            if ok.sum() < min_trades:
                continue
            capacity = n_symbols * duration_s / (h + 1.0)  # non-overlapping positions possible
            qs = np.unique(np.quantile(s[ok], QUANTILES))
            for th in qs:
                sel = ok & (s >= th)
                n = int(sel.sum())
                if n < min_trades:
                    continue
                mean, se, g = cluster_mean_se(r[sel], clusters[sel])
                lcb = mean - z * se
                obj = min(n, capacity) * lcb
                cands.append({"h": h, "th": float(th), "n": n, "mean": mean, "lcb": lcb, "clusters": g})
                if lcb > min_lcb and obj > best_obj:
                    best_obj, best = obj, (h, float(th), sel, mean, se, lcb)
        if best is None:
            sides[name] = None
            rejected[name] = "no threshold with positive cluster-robust LCB on calibration data"
            continue
        h, th, sel, mean, se, lcb = best
        r = cal[TargetSpec.col(name, h)].to_numpy(float)
        s = scores[(name, h)]
        ok = ~np.isnan(r) & np.isfinite(s)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s[ok], r[ok]) if ok.sum() > 50 else None
        rs = r[sel]
        mae = cal[f"mae_{name}_{h:g}"].to_numpy(float)[sel]
        slip = cal[f"slip_bps_{name}"].to_numpy(float)[sel]
        sp = SidePolicy(
            side=name, horizon=h, threshold=th, n_cal=int(sel.sum()), mean=float(mean), se=float(se), lcb=float(lcb),
            std=float(rs.std(ddof=1)), p_pos=float((rs > 0).mean()), q05=float(np.quantile(rs, 0.05)),
            q95=float(np.quantile(rs, 0.95)), es05=expected_shortfall(rs, 0.05), mae_mean=float(np.nanmean(mae)),
            p_sig_slip=float(np.nanmean(slip > sig_slip_bps)),
            hold_ms=float(np.nanmean(cal[f"hold_ms_{h:g}"].to_numpy(float)[sel])),
            candidates=sorted(cands, key=lambda c: -c["lcb"])[:10], _iso=iso,
        )
        if passive and f"pfill_{name}" in cal:
            pf = cal[f"pfill_{name}"].to_numpy(float)[sel]
            pr = cal[f"pret_{name}_{h:g}"].to_numpy(float)[sel]
            filled = pf == 1.0
            sp.passive_fill = float(np.nanmean(pf)) if np.isfinite(pf).any() else math.nan
            if filled.sum() >= 10:
                m2, se2, _ = cluster_mean_se(pr[filled & ~np.isnan(pr)], clusters[sel][filled & ~np.isnan(pr)])
                sp.passive_mean_if_filled, sp.passive_se = float(m2), float(se2)
        sides[name] = sp
    return Policy(model_name, sides, rejected)

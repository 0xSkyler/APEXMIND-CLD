"""Regularized linear, gradient-boosted and (gated) temporal models.

All predict net executable return per (side, horizon) from features chosen
by :func:`select_features` on the training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from apexmind.features.selection import SelectionResult, select_features
from apexmind.models.base import Model, TargetSpec

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _select(train: pd.DataFrame, features: list[str], spec: TargetSpec, enabled: bool) -> SelectionResult:
    if not enabled:
        return SelectionResult(list(features), {}, {})
    targets = {f"{s}_{h:g}": train[TargetSpec.col(s, h)].to_numpy(float) for s, h in spec.pairs()}
    return select_features(train[features], targets)


class _Standardizer:
    def fit(self, X: np.ndarray) -> "_Standardizer":
        self.med = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xi.mean(axis=0), Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = np.where(np.isnan(X), self.med, X)
        return np.clip((Xi - self.mu) / self.sd, -8, 8)


class RidgeModel(Model):
    name, family, complexity = "ridge", "linear", 4

    def __init__(self, features: list[str], select: bool = True) -> None:
        self.features, self.do_select = list(features), select

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.sel = _select(train, self.features, spec, self.do_select)
        self.cols = self.sel.selected
        self.models, self.alpha = {}, {}
        if not self.cols:
            return
        X = train[self.cols].to_numpy(float)
        self.std = _Standardizer().fit(X)
        Z = self.std.transform(X)
        cut = int(len(Z) * 0.75)
        for s, h in spec.pairs():
            y = train[TargetSpec.col(s, h)].to_numpy(float)
            m = ~np.isnan(y)
            tr, va = m.copy(), m.copy()
            tr[cut:], va[:cut] = False, False
            best, best_err = ALPHAS[-1], np.inf
            if tr.sum() > 100 and va.sum() > 100:
                for a in ALPHAS:
                    r = Ridge(alpha=a).fit(Z[tr], y[tr])
                    err = np.mean((r.predict(Z[va]) - y[va]) ** 2)
                    if err < best_err:
                        best, best_err = a, err
            self.alpha[(s, h)] = best
            self.models[(s, h)] = Ridge(alpha=best).fit(Z[m], y[m])

    def score(self, df: pd.DataFrame) -> dict:
        if not self.models:
            return {p: np.full(len(df), -np.inf) for p in self.spec.pairs()}
        Z = self.std.transform(df[self.cols].to_numpy(float))
        return {p: m.predict(Z) for p, m in self.models.items()}

    def describe(self) -> dict:
        coefs = {f"{s}_{h:g}": dict(zip(self.cols, np.round(m.coef_ * 1e4, 4).tolist())) for (s, h), m in self.models.items()}
        return {**super().describe(), "selection": self.sel.to_dict(),
                "alpha": {f"{s}_{h:g}": a for (s, h), a in self.alpha.items()}, "coef_bps_per_sd": coefs}


def _gbm(n: int, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=max(200, n // 200), l2_regularization=1.0, early_stopping=False, random_state=seed)


class GBMModel(Model):
    name, family, complexity = "gbm", "gbm", 5

    def __init__(self, features: list[str], select: bool = True, seed: int = 0) -> None:
        self.features, self.do_select, self.seed = list(features), select, seed

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.cols].to_numpy(np.float32)

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.sel = _select(train, self.features, spec, self.do_select)
        self.cols = self.sel.selected
        self.models = {}
        if not self.cols:
            return
        X = self._design(train)
        for s, h in spec.pairs():
            y = train[TargetSpec.col(s, h)].to_numpy(float)
            m = ~np.isnan(y)
            if m.sum() < 500:
                continue
            self.models[(s, h)] = _gbm(int(m.sum()), self.seed).fit(X[m], y[m])

    def score(self, df: pd.DataFrame) -> dict:
        out = {}
        X = self._design(df) if self.models else None
        for p in self.spec.pairs():
            out[p] = self.models[p].predict(X) if p in self.models else np.full(len(df), -np.inf)
        return out

    def describe(self) -> dict:
        return {**super().describe(), "selection": self.sel.to_dict(), "fitted": [f"{s}_{h:g}" for s, h in self.models]}


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    """Lagged copies of ``cols`` within each symbol (rows are grid ticks)."""
    out = {}
    g = df.groupby("symbol", observed=True, sort=False)
    for k in lags:
        shifted = g[cols].shift(k)
        for c in cols:
            out[f"{c}__lag{k}"] = shifted[c].to_numpy()
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


class TemporalGBM(GBMModel):
    """GBM on current plus lagged feature values.

    Lag columns (``<feature>__lag<k>``) must be precomputed on the full,
    contiguous grid with :func:`add_lags` before any row thinning.

    Gate: it is only *justified* if, on a time holdout inside the training
    window, adding lags reduces squared error by a margin that is
    significant under a cluster bootstrap. Otherwise it never trades and the
    research report lists it as gated out.
    """

    name, family, complexity = "temporal_gbm", "temporal", 6
    LAGS = (4, 20)  # grid ticks: 1 s and 5 s at the default 250 ms grid

    def __init__(self, features: list[str], select: bool = True, seed: int = 0, cluster_rows: int = 600) -> None:
        super().__init__(features, select, seed)
        self.cluster_rows = cluster_rows
        self.justified, self.gate = False, {}

    @classmethod
    def lag_columns(cls, cols: list[str]) -> list[str]:
        return [f"{c}__lag{k}" for k in cls.LAGS for c in cols]

    def fit(self, train: pd.DataFrame, spec: TargetSpec) -> None:
        self.spec = spec
        self.sel = _select(train, self.features, spec, self.do_select)
        base_cols = self.sel.selected
        self.models = {}
        if not base_cols:
            self.gate = {"reason": "no selected features"}
            return
        lag_cols = base_cols + self.lag_columns(base_cols)
        missing = [c for c in lag_cols if c not in train.columns]
        if missing:
            raise ValueError(f"lag columns not precomputed: {missing[:3]}...")
        cut = int(len(train) * 0.75)
        Xb, Xl = train[base_cols].to_numpy(np.float32), train[lag_cols].to_numpy(np.float32)
        improvements = []
        for s, h in spec.pairs():
            y = train[TargetSpec.col(s, h)].to_numpy(float)
            m = ~np.isnan(y)
            tr, va = m.copy(), m.copy()
            tr[cut:], va[:cut] = False, False
            if tr.sum() < 1000 or va.sum() < 500:
                continue
            eb = (_gbm(int(tr.sum()), self.seed).fit(Xb[tr], y[tr]).predict(Xb[va]) - y[va]) ** 2
            el = (_gbm(int(tr.sum()), self.seed).fit(Xl[tr], y[tr]).predict(Xl[va]) - y[va]) ** 2
            improvements.append(eb - el)
        if not improvements:
            self.gate = {"reason": "insufficient data for gate"}
            return
        n = min(map(len, improvements))
        d = np.mean(np.vstack([x[:n] for x in improvements]), axis=0)
        n_cl = max(2, len(d) // self.cluster_rows)
        means = np.array([c.mean() for c in np.array_split(d, n_cl)])
        rng = np.random.default_rng(self.seed)
        boots = rng.choice(means, (2000, len(means))).mean(axis=1)
        p_value = float((boots <= 0).mean())
        self.justified = bool(p_value < 0.05 and d.mean() > 0)
        self.gate = {"mse_improvement": float(d.mean()), "p_value": p_value, "clusters": int(n_cl),
                     "justified": self.justified}
        if not self.justified:
            return
        self.cols = lag_cols
        for s, h in spec.pairs():
            y = train[TargetSpec.col(s, h)].to_numpy(float)
            m = ~np.isnan(y)
            if m.sum() >= 500:
                self.models[(s, h)] = _gbm(int(m.sum()), self.seed).fit(Xl[m], y[m])

    def describe(self) -> dict:
        return {**super().describe(), "gate": self.gate}


def make_model(name: str, features: list[str], select: bool = True, seed: int = 0) -> Model:
    from apexmind.models.baselines import FlowOnly, LagFlow, LagOnly, RefMomentum, ShockReversion, VolMomentum

    table = {
        "ref_momentum": lambda: RefMomentum(),
        "lag_only": lambda: LagOnly(),
        "flow_only": lambda: FlowOnly(),
        "lag_flow": lambda: LagFlow(),
        "ridge": lambda: RidgeModel(features, select),
        "gbm": lambda: GBMModel(features, select, seed),
        "temporal_gbm": lambda: TemporalGBM(features, select, seed),
        "shock_reversion": lambda: ShockReversion(),
        "vol_momentum": lambda: VolMomentum(),
    }
    if name not in table:
        raise ValueError(f"unknown model {name}")
    return table[name]()

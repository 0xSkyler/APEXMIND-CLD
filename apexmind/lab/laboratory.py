"""Alpha Laboratory: continuous, resource-controlled research with
champion/challenger governance.

Each cycle:

1. build (or load cached) datasets over the recent lookback window;
2. run every active hypothesis family through the walk-forward pipeline;
3. re-run the research to verify the results hash is reproducible;
4. apply the promotion gate (vs. baselines and vs. the current champion);
5. if promoted, fit a production bundle on the newest data and register it;
6. monitor the current champion on data it has not seen since promotion and
   retire it if its edge is gone;
7. measure feature drift (PSI) between recent and older data.

Secondary strategy families only run once the primary family (CVIFA) has a
credible champion, and must pass the same gate on their own.
"""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from apexmind.backtest.simulator import simulate
from apexmind.collector.recorder import data_span
from apexmind.config import Config
from apexmind.data.dataset import DatasetMeta, build_dataset
from apexmind.lab.promotion import promotion_gate
from apexmind.lab.registry import Registry
from apexmind.models.base import TargetSpec, training_rows
from apexmind.models.ml import TemporalGBM, add_lags, make_model
from apexmind.report.proof import write_report
from apexmind.research.evaluate import period_series, trade_metrics
from apexmind.research.pipeline import purge_gap_ns, run_research
from apexmind.research.policy import fit_policy
from apexmind.research.splits import window
from apexmind.research.stats import paired_difference
from apexmind.strategy.decision import StrategyBundle
from apexmind.util import atomic_write_json, config_hash, git_revision
from apexmind.venues.lighter.markets import LighterMarket

log = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    name: str
    models: list[str]
    primary: bool = False


def hypotheses(cfg: Config) -> list[Hypothesis]:
    return [
        Hypothesis("cvifa", list(cfg.research.models), primary=True),
        Hypothesis("liquidity_shock_reversion", ["ref_momentum", "lag_only", "shock_reversion"]),
        Hypothesis("volatility_regime_momentum", ["ref_momentum", "lag_only", "vol_momentum"]),
    ]


def apply_resource_limits(nice: int, max_memory_gb: float, max_threads: int) -> None:
    """Lower priority, cap address space and native thread pools."""
    try:
        os.nice(max(0, nice - os.nice(0)))
    except OSError:
        pass
    limit = int(max_memory_gb * 1.5 * 2**30)  # virtual address space headroom over RSS budget
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or limit < hard:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ValueError, OSError):
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(max_threads)
    except ImportError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``actual`` vs ``expected``."""
    e = expected[np.isfinite(expected)]
    a = actual[np.isfinite(actual)]
    if len(e) < 100 or len(a) < 100:
        return math.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return math.nan
    pe = np.histogram(np.clip(e, edges[0], edges[-1]), edges)[0] / len(e) + 1e-6
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(df: pd.DataFrame, features: list[str], recent_frac: float = 0.2) -> dict[str, float]:
    ts = df["ts_ns"].to_numpy()
    cut = np.quantile(ts, 1 - recent_frac)
    old, new = df[ts < cut], df[ts >= cut]
    out = {f: psi(old[f].to_numpy(float), new[f].to_numpy(float)) for f in features if f in df}
    return dict(sorted(((k, round(v, 4)) for k, v in out.items() if v == v), key=lambda kv: -kv[1])[:15])


def fit_production_bundle(cfg: Config, df: pd.DataFrame, meta: DatasetMeta, model_name: str, mode: str,
                          manifest: dict) -> StrategyBundle:
    """Fit on the newest data: train window, purge gap, calibration window."""
    spec = TargetSpec(tuple(cfg.labels.horizons_s))
    features = list(meta.feature_names)
    if model_name == "temporal_gbm":
        df = add_lags(df.sort_values(["symbol", "ts_ns"]), features, TemporalGBM.LAGS).sort_values(["ts_ns", "symbol"])
    gap = purge_gap_ns(cfg, meta.latency["p99_ms"])
    t0, t1 = int(df.ts_ns.min()), int(df.ts_ns.max()) + 1
    cal_start = t1 - int((t1 - t0) * cfg.research.calibration_fraction)
    fit = training_rows(window(df, t0, cal_start - gap), spec, cfg.research.max_train_rows)
    cal = window(df, cal_start, t1)
    cal = cal[cal["valid"]]
    model = make_model(model_name, features, cfg.research.feature_selection, cfg.research.seed)
    model.fit(fit, spec)
    pol = fit_policy(model_name, model.score(cal), cal, spec, z=cfg.strategy.lcb_z, min_lcb=cfg.strategy.min_lcb_return,
                     block_ns=int(cfg.research.block_minutes * 60e9), min_trades=cfg.research.min_calibration_trades,
                     n_symbols=len(meta.symbols), sig_slip_bps=cfg.labels.significant_slippage_bps,
                     passive=cfg.labels.passive)
    if not cfg.execution.allow_passive or mode == "aggressive":
        for sp in pol.sides.values():
            if sp is not None:
                sp.passive_fill = math.nan  # optimizer then always chooses aggressive
    g = cal.groupby("symbol", observed=True)
    spreads = g["lit_spread_bps"].median().to_dict()
    impact = (0.5 * (g["lit_impact_buy_bps"].median() + g["lit_impact_sell_bps"].median())).to_dict()
    return StrategyBundle(
        name=model_name, model=model, policy=pol, feature_names=features,
        calib_spread_bps={str(k): float(v) for k, v in spreads.items()},
        calib_impact_bps={str(k): float(v) for k, v in impact.items()},
        calib_latency_ms=float(meta.latency["p50_ms"]),
        manifest={**manifest, "model": model_name, "mode": mode, "policy": pol.to_dict(),
                  "fit_window": [t0, cal_start - gap], "cal_window": [cal_start, t1],
                  "data_kind": meta.data_kind, "latency": meta.latency, "data_fingerprint": meta.data_fingerprint,
                  "config_hash": config_hash(cfg), "git": git_revision(), "markets": meta.markets},
    )


def evaluate_bundle(bundle: StrategyBundle, df: pd.DataFrame, meta: DatasetMeta, cfg: Config,
                    initial_equity: float = 1000.0) -> dict:
    """Out-of-sample check of a registered bundle on data it never saw."""
    test = df[df["valid"]].sort_values(["ts_ns", "symbol"]).reset_index(drop=True)
    if test.empty:
        return {"n_trades": 0}
    dec = bundle.policy.decide(bundle.model.score(test), len(test))
    markets = {s: LighterMarket(**m) for s, m in meta.markets.items()}
    mode = bundle.manifest.get("mode", "aggressive")
    bt = simulate(test, dec, bundle.policy, markets, cfg, initial_equity, mode)
    m = trade_metrics(bt.trades, int(cfg.research.block_minutes * 60e9), 500, cfg.research.confidence)
    m["series"] = period_series(bt.trades, int(test.ts_ns.min()), int(test.ts_ns.max()) + 1,
                                int(cfg.research.block_minutes * 60e9)).tolist()
    return m


class AlphaLab:
    def __init__(self, cfg: Config, data_dir: str, purpose: str = "live") -> None:
        self.cfg, self.data_dir, self.purpose = cfg, data_dir, purpose
        self.registry = Registry(cfg.lab.registry_dir)
        self.runs = Path(cfg.research.runs_dir)
        self.cache = self.runs / "cache"

    def _record(self, rec: dict) -> None:
        self.runs.mkdir(parents=True, exist_ok=True)
        with open(self.runs / "experiments.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def run_cycle(self, lookback_hours: float | None = None) -> dict:
        span = data_span(self.data_dir, ["lighter", self.cfg.reference.venue])
        rc = self.cfg.research
        need_h = rc.min_train_hours + rc.test_hours + 1.0  # one fold plus embargo gaps
        have_h = (span[1] - span[0]) / 3.6e12 if span else 0.0
        if have_h < need_h:
            # Written every check so operators can see progress instead of "missing".
            status = {"t": time.time(), "status": "waiting_for_data", "recorded_hours": round(have_h, 2),
                      "first_run_at_hours": need_h,
                      "full_protocol_hours": rc.min_train_hours + rc.n_folds * rc.test_hours}
            atomic_write_json(self.runs / "lab_status.json", status)
            return status
        start, end = span
        if lookback_hours:
            start = max(start, end - int(lookback_hours * 3.6e12))
        df, meta = build_dataset(self.cfg, self.data_dir, start, end, cache_dir=str(self.cache))
        cycle = {"t": time.time(), "period": [start, end], "data_kind": meta.data_kind, "results": {}}
        champion = self.registry.load_champion()
        cycle["drift_psi"] = feature_drift(df, meta.feature_names)
        if champion is not None:
            cycle["champion_check"] = self._monitor_champion(champion, df, meta)
        primary_credible = champion is not None and champion[1].manifest.get("hypothesis") == "cvifa"
        for hyp in hypotheses(self.cfg):
            if not hyp.primary and not primary_credible:
                cycle["results"][hyp.name] = {"status": "deferred: primary strategy has no credible champion"}
                continue
            run_dir = self.runs / f"{hyp.name}-{time.strftime('%Y%m%dT%H%M%S')}"
            res = run_research(self.cfg, df, meta, run_dir, hyp.models)
            write_report(run_dir)
            rerun = run_research(self.cfg, df, meta, run_dir / "reproduction", hyp.models)
            sel = res["champion"].get("selected")
            vs_champ = self._vs_champion(champion, res, sel, df, meta) if champion and sel else None
            gate = promotion_gate(res, self.cfg, self.purpose, rerun["meta"]["results_hash"], vs_champ,
                                  champion_exists=champion is not None)
            entry = {"run_dir": str(run_dir), "selected": sel, "gate": gate.to_dict()}
            if gate.passed:
                bundle = fit_production_bundle(self.cfg, df, meta, sel["model"], sel["mode"],
                                               {"hypothesis": hyp.name, "research_run": str(run_dir),
                                                "results_hash": res["meta"]["results_hash"], "gate": gate.to_dict(),
                                                "purpose": self.purpose})
                bid = self.registry.save(bundle)
                self.registry.promote(bid, gate.to_dict())
                entry["promoted"] = bid
            cycle["results"][hyp.name] = entry
        self._record(cycle)
        atomic_write_json(self.runs / "lab_status.json", cycle)
        return cycle

    def _vs_champion(self, champion, res: dict, sel: dict, df: pd.DataFrame, meta: DatasetMeta) -> dict | None:
        """Paired comparison on OOS periods that the champion never saw."""
        _, bundle = champion
        ser = res["series"][f"{sel['model']}/{sel['mode']}"]
        period_ns, p0 = ser["period_ns"], ser["start_period"]
        mine = pd.Series(ser["values"], index=np.arange(p0, p0 + len(ser["values"])))
        unseen_from = bundle.manifest.get("cal_window", [0, 0])[1] // period_ns + 1
        mine = mine[mine.index >= unseen_from]
        if len(mine) < 5:
            return None
        sub = window(df, int(mine.index[0] * period_ns), int((mine.index[-1] + 1) * period_ns))
        ev = evaluate_bundle(bundle, sub, meta, self.cfg)
        theirs = pd.Series(ev.get("series", []), index=np.arange(mine.index[0], mine.index[0] + len(ev.get("series", []))))
        both = pd.concat([mine.rename("a"), theirs.rename("b")], axis=1).fillna(0.0)
        return paired_difference(both["a"].to_numpy(), both["b"].to_numpy(), self.cfg.research.bootstrap_reps,
                                 self.cfg.research.confidence, self.cfg.research.seed)

    def _monitor_champion(self, champion: tuple[str, StrategyBundle], df: pd.DataFrame, meta: DatasetMeta) -> dict:
        bid, bundle = champion
        seen_until = bundle.manifest.get("cal_window", [0, 0])[1]
        new = df[df["ts_ns"] >= seen_until]
        if new.empty:
            return {"id": bid, "status": "no unseen data yet"}
        m = evaluate_bundle(bundle, new, meta, self.cfg)
        m.pop("series", None)
        n = m.get("n_trades", 0)
        hi = (m.get("mean_net_bps_ci") or [None, None])[1]
        status = "ok"
        if n >= self.cfg.risk.edge_min_trades and hi is not None and hi < 0:
            self.registry.retire(f"edge disappeared on unseen data: net CI upper {hi} bps over {n} trades")
            status = "retired"
        return {"id": bid, "status": status, "unseen_metrics": m}

    def run_forever(self, lookback_hours: float | None = None) -> None:
        lab = self.cfg.lab
        apply_resource_limits(lab.nice, lab.max_memory_gb, lab.max_threads)
        while True:
            t0 = time.time()
            try:
                out = self.run_cycle(lookback_hours)
                log.info("lab cycle done: %s", {k: v for k, v in out.items() if k != "results"})
            except Exception:
                log.error("lab cycle failed:\n%s", traceback.format_exc())
                self._record({"t": time.time(), "error": traceback.format_exc()})
                atomic_write_json(self.runs / "lab_status.json",
                                  {"t": time.time(), "status": "error", "error": traceback.format_exc()[-2000:]})
                out = {}
            # While waiting for data, re-check every 30 min so the first run starts soon after it is possible.
            interval = 30.0 if out.get("status") == "waiting_for_data" else lab.interval_minutes
            time.sleep(max(60.0, interval * 60 - (time.time() - t0)))

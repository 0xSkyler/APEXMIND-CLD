"""Walk-forward research run: every model, identical chronological windows,
calibrated uncertainty-aware policies, execution-aware simulation, and the
statistics needed to decide whether anything beats simple baselines after
costs."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from apexmind import __version__
from apexmind.backtest.simulator import simulate
from apexmind.config import Config
from apexmind.data.dataset import DatasetMeta, summarize_period
from apexmind.models.base import TargetSpec, training_rows
from apexmind.models.ml import TemporalGBM, add_lags, make_model
from apexmind.research.evaluate import (
    equity_metrics,
    period_series,
    regime_breakdown,
    rejection_summary,
    trade_metrics,
)
from apexmind.research.policy import QUANTILES, fit_policy
from apexmind.research.regimes import RegimeModel
from apexmind.research.splits import make_folds, window
from apexmind.research.stats import deflated_sharpe, paired_difference, pbo_cscv, sharpe, spa_test
from apexmind.util import atomic_write_json, config_hash, git_revision, stable_hash
from apexmind.venues.lighter.markets import LighterMarket

log = logging.getLogger(__name__)

BASELINES = ("ref_momentum", "lag_only", "flow_only", "lag_flow")
MODES = ("aggressive", "optimizer")


def purge_gap_ns(cfg: Config, latency_p99_ms: float) -> int:
    span = max(cfg.labels.horizons_s) + 2 * latency_p99_ms / 1e3 + cfg.labels.passive_wait_s + 10.0
    return int(max(cfg.research.embargo_s, span) * 1e9)


def run_research(cfg: Config, df: pd.DataFrame, meta: DatasetMeta, run_dir: str | Path,
                 models: list[str] | None = None) -> dict:
    t_start = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    models = list(models or cfg.research.models)
    spec = TargetSpec(tuple(cfg.labels.horizons_s))
    features = list(meta.feature_names)
    df = df.sort_values(["symbol", "ts_ns"]).reset_index(drop=True)
    if "temporal_gbm" in models:
        df = add_lags(df, features, TemporalGBM.LAGS)  # on the contiguous per-symbol grid
    df = df.sort_values(["ts_ns", "symbol"], kind="stable").reset_index(drop=True)
    gap = purge_gap_ns(cfg, meta.latency["p99_ms"])
    rc = cfg.research
    folds = make_folds(int(df.ts_ns.min()), int(df.ts_ns.max()) + 1, rc.n_folds, rc.min_train_hours, rc.test_hours,
                       rc.calibration_fraction, gap)
    markets = {s: LighterMarket(**m) for s, m in meta.markets.items()}
    block_ns = int(rc.block_minutes * 60e9)
    equities = rc.initial_equities
    acc: dict = {m: {"policies": [], "describe": [], "gates": [], "folds": {mode: [] for mode in MODES},
                     "trades": {(mode, e): [] for mode in MODES for e in equities},
                     "curves": {(mode, e): [] for mode in MODES for e in equities},
                     "rej": {(mode, e): Counter() for mode in MODES for e in equities},
                     "signals": {(mode, e): 0 for mode in MODES for e in equities},
                     "equity": {(mode, e): e for mode in MODES for e in equities}} for m in models}
    test_span = (folds[0].test_start, folds[-1].test_end)
    for fold in folds:
        fit = training_rows(window(df, fold.fit_start, fold.fit_end), spec, rc.max_train_rows)
        cal = window(df, fold.cal_start, fold.cal_end)
        cal = cal[cal["valid"]]
        test = window(df, fold.test_start, fold.test_end)
        test = test[test["valid"]].reset_index(drop=True)
        regimes = RegimeModel.fit(fit)
        log.info("fold %d: fit=%d cal=%d test=%d rows", fold.index, len(fit), len(cal), len(test))
        for name in models:
            model = make_model(name, features, rc.feature_selection, rc.seed)
            model.fit(fit, spec)
            info = model.describe()
            acc[name]["describe"].append({"fold": fold.index, **info})
            if isinstance(model, TemporalGBM) and not model.justified:
                acc[name]["gates"].append({"fold": fold.index, **model.gate})
                continue
            pol = fit_policy(name, model.score(cal), cal, spec, z=cfg.strategy.lcb_z,
                             min_lcb=cfg.strategy.min_lcb_return, block_ns=block_ns,
                             min_trades=rc.min_calibration_trades, n_symbols=len(meta.symbols),
                             sig_slip_bps=cfg.labels.significant_slippage_bps, passive=cfg.labels.passive)
            acc[name]["policies"].append({"fold": fold.index, **pol.to_dict()})
            dec = pol.decide(model.score(test), len(test))
            for mode in MODES:
                for e in equities:
                    key = (mode, e)
                    bt = simulate(test, dec, pol, markets, cfg, acc[name]["equity"][key], mode)
                    acc[name]["equity"][key] = bt.final_equity
                    acc[name]["signals"][key] += bt.n_signals
                    acc[name]["rej"][key] += bt.rejections
                    acc[name]["curves"][key].append(bt.equity_curve)
                    if not bt.trades.empty:
                        tr = bt.trades.assign(fold=fold.index, model=name)
                        reg = regimes.label(tr["ref_rv_bps"].to_numpy(), tr["lit_spread_bps"].to_numpy(),
                                            tr["trend_60"].to_numpy())
                        acc[name]["trades"][key].append(pd.concat([tr, reg], axis=1))
                    if e == equities[0]:
                        fm = trade_metrics(bt.trades, block_ns, 500, rc.confidence, rc.seed)
                        fm.update({"fold": fold.index, "signals": bt.n_signals,
                                   "return": round(bt.final_equity / bt.initial_equity - 1, 6), "notes": bt.notes})
                        acc[name]["folds"][mode].append(fm)

    # ---- aggregate -----------------------------------------------------------
    period_ns = block_ns
    results: dict = {"models": {}, "folds": [f.to_dict() for f in folds]}
    series: dict[tuple[str, str], pd.Series] = {}
    all_trades = []
    for name in models:
        a = acc[name]
        res = {"describe": a["describe"], "policies": a["policies"], "gates": a["gates"], "modes": {}}
        for mode in MODES:
            mres = {"per_fold": a["folds"][mode], "equity_scenarios": {}}
            for e in equities:
                key = (mode, e)
                trades = pd.concat(a["trades"][key], ignore_index=True) if a["trades"][key] else pd.DataFrame()
                curve = (pd.concat(a["curves"][key], ignore_index=True) if a["curves"][key]
                         else pd.DataFrame({"ts_ns": [0], "equity": [e]}))
                scen = {"trades": trade_metrics(trades, block_ns, rc.bootstrap_reps, rc.confidence, rc.seed),
                        "equity": equity_metrics(curve, trades, e, period_ns),
                        "opportunities": rejection_summary(a["rej"][key], a["signals"][key], len(trades))}
                if e == equities[0]:
                    scen["regimes"] = regime_breakdown(trades, ["vol_regime", "spread_regime", "trend_regime"],
                                                       block_ns, rc.confidence)
                    series[(name, mode)] = period_series(trades, test_span[0], test_span[1], period_ns)
                    if not trades.empty:
                        all_trades.append(trades.assign(mode=mode, initial_equity=e))
                mres["equity_scenarios"][f"{e:g}"] = scen
            res["modes"][mode] = mres
        results["models"][name] = res

    results["series"] = {f"{m}/{mode}": {"start_period": int(v.index[0]) if len(v) else 0,
                                         "period_ns": period_ns, "values": v.round(10).tolist()}
                         for (m, mode), v in series.items()}
    results["comparisons"] = compare(series, models, cfg, len(spec.horizons))
    results["champion"] = choose_champion(results, series, models, folds, cfg)
    results["meta"] = {
        "version": __version__, "git": git_revision(), "config_hash": config_hash(cfg),
        "dataset": {**{k: v for k, v in meta.to_dict().items() if k not in ("quality", "markets")},
                    "period": summarize_period(df)},
        "data_quality": meta.quality, "markets": meta.markets, "gap_s": gap / 1e9, "elapsed_s": round(time.time() - t_start, 1),
        "trials": len(models) * 2 * len(spec.horizons) * len(QUANTILES),
    }
    results["meta"]["results_hash"] = stable_hash({k: results[k] for k in ("models", "comparisons", "champion")})
    atomic_write_json(run_dir / "results.json", results)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(run_dir / "trades.csv", index=False)
    return results


def compare(series: dict, models: list[str], cfg: Config, n_h: int) -> dict:
    rc = cfg.research
    out: dict = {}
    for mode in MODES:
        avail = [m for m in models if (m, mode) in series]
        if not avail:
            continue
        mat = pd.concat([series[(m, mode)].rename(m) for m in avail], axis=1).fillna(0.0)
        cmp: dict = {"periods": int(len(mat)), "pairwise_vs_baselines": {}}
        base = [m for m in avail if m in BASELINES]
        for m in avail:
            cmp["pairwise_vs_baselines"][m] = {
                b: paired_difference(mat[m].to_numpy(), mat[b].to_numpy(), rc.bootstrap_reps, rc.confidence, rc.seed)
                for b in base if b != m}
        spa0 = spa_test(mat.to_numpy(), rc.bootstrap_reps, seed=rc.seed)
        cmp["spa_vs_no_trading"] = {"p_value": spa0.p_value, "best_model": avail[spa0.best] if spa0.best >= 0 else None}
        non_base = [m for m in avail if m not in BASELINES]
        if base and non_base:
            best_base = max(base, key=lambda b: mat[b].mean())
            d = mat[non_base].to_numpy() - mat[[best_base]].to_numpy()
            spa1 = spa_test(d, rc.bootstrap_reps, seed=rc.seed)
            cmp["spa_vs_best_baseline"] = {"benchmark": best_base, "p_value": spa1.p_value,
                                           "best_model": non_base[spa1.best] if spa1.best >= 0 else None}
        srs = [sharpe(mat[m].to_numpy()) for m in avail]
        sr_var = float(np.nanvar(srs)) if len(avail) > 1 else 0.0
        trials = len(models) * 2 * n_h * len(QUANTILES)
        cmp["deflated_sharpe"] = {m: deflated_sharpe(mat[m].to_numpy(), trials, sr_var) for m in avail}
        n_splits = max(2, min(8, (len(mat) // 2) * 2))  # CSCV needs an even number of groups
        cmp["pbo"] = pbo_cscv(mat.to_numpy(), n_splits=n_splits) if len(avail) > 1 else {}
        cmp["mean_period_net"] = {m: float(mat[m].mean()) for m in avail}
        out[mode] = cmp
    return out


def choose_champion(results: dict, series: dict, models: list[str], folds, cfg: Config) -> dict:
    """Simplest model whose OOS edge is credible and not significantly worse
    than the best credible model. Returns a verdict with reasons."""
    rc = cfg.research
    complexity = {m: make_model(m, [], False).complexity for m in models}
    verdicts = {}
    eligible = []
    for m in models:
        for mode in MODES:
            mres = results["models"][m]["modes"][mode]
            scen = mres["equity_scenarios"][f"{rc.initial_equities[0]:g}"]
            t = scen["trades"]
            reasons = []
            if t.get("n_trades", 0) < 30:
                reasons.append(f"only {t.get('n_trades', 0)} OOS trades")
            else:
                lo = t["mean_net_bps_ci"][0]
                if lo is None or lo <= 0:
                    reasons.append(f"net mean CI lower bound {lo} bps <= 0")
                pf = t.get("profit_factor")
                if pf is not None and pf <= 1:
                    reasons.append(f"profit factor {pf} <= 1")
                pos_folds = sum(1 for f in mres["per_fold"] if (f.get("mean_net_bps") or 0) > 0)
                if pos_folds < math.ceil(len(folds) / 2 + 0.01):
                    reasons.append(f"positive in only {pos_folds}/{len(folds)} OOS periods")
            verdicts[f"{m}/{mode}"] = reasons or ["credible"]
            if not reasons:
                eligible.append((m, mode))
    if not eligible:
        return {"selected": None, "verdict": "NO MODEL demonstrated a credible out-of-sample edge after costs",
                "checks": verdicts}
    best = max(eligible, key=lambda k: series[k].mean())
    chosen = best
    for k in sorted(eligible, key=lambda k: (complexity[k[0]], k[1] != "aggressive")):
        if k == best:
            break
        d = paired_difference(series[best].to_numpy(), series[k].to_numpy(), rc.bootstrap_reps, rc.confidence, rc.seed)
        if d["p_value"] > 0.05:  # best is not significantly better: prefer the simpler one
            chosen = k
            break
    beats = {}
    for b in BASELINES:
        if b in models and b != chosen[0] and (b, chosen[1]) in series:
            beats[b] = paired_difference(series[chosen].to_numpy(), series[(b, chosen[1])].to_numpy(),
                                         rc.bootstrap_reps, rc.confidence, rc.seed)
    beats_all = bool(beats) and all(v["p_value"] < 0.05 and v["mean"] > 0 for v in beats.values())
    return {"selected": {"model": chosen[0], "mode": chosen[1]}, "best_by_mean": {"model": best[0], "mode": best[1]},
            "beats_all_baselines_after_costs": beats_all, "vs_baselines": beats, "checks": verdicts,
            "verdict": ("credible OOS edge; " + ("beats every baseline" if beats_all else
                        "does NOT significantly beat every baseline"))}

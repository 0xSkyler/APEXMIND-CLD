"""Out-of-sample performance metrics for a model's trades."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from apexmind.research.stats import (
    cluster_bootstrap,
    cluster_mean_se,
    expected_shortfall,
    max_drawdown,
    profit_factor,
    time_clusters,
)


def _bps(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x * 1e4, 3)


def trade_metrics(trades: pd.DataFrame, block_ns: int, reps: int, conf: float, seed: int = 0) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    r = trades["ret"].to_numpy(float)
    cl = time_clusters(trades["t_entry"].to_numpy(), block_ns)
    mean, lo, hi = cluster_bootstrap(r, cl, np.mean, reps, conf, seed)
    _, se, n_cl = cluster_mean_se(r, cl)
    pnl = trades["pnl"].to_numpy(float)
    gross = trades["gross"].to_numpy(float)
    return {
        "n_trades": int(len(r)),
        "n_time_blocks": int(n_cl),
        "mean_net_bps": _bps(mean),
        "mean_net_bps_ci": [_bps(lo), _bps(hi)],
        "mean_net_bps_cluster_se": _bps(se),
        "median_net_bps": _bps(float(np.median(r))),
        "hit_rate": round(float((r > 0).mean()), 4),
        "mean_gross_bps": _bps(float(np.nanmean(gross))) if np.isfinite(gross).any() else None,
        "mean_fees_bps": _bps(float(trades["fees"].mean())),
        "mean_funding_bps": _bps(float(trades["funding"].mean())),
        "mean_entry_slippage_bps": round(float(trades["slip_bps"].mean()), 3),
        "p90_entry_slippage_bps": round(float(trades["slip_bps"].quantile(0.9)), 3),
        "mean_mae_bps": _bps(float(trades["mae"].mean())),
        "latency_ms_p50": round(float(trades["lat_entry_ms"].median()), 1),
        "latency_ms_p90": round(float(trades["lat_entry_ms"].quantile(0.9)), 1),
        "profit_factor": round(profit_factor(r), 4) if np.isfinite(profit_factor(r)) else None,
        "expectancy_usd": round(float(pnl.mean()), 6),
        "expected_shortfall_95_bps": _bps(expected_shortfall(r, 0.05)),
        "worst_trade_bps": _bps(float(r.min())),
        "best_trade_bps": _bps(float(r.max())),
        "passive_share": round(float((trades["exec_mode"] == "passive").mean()), 4),
        "mean_predicted_bps": _bps(float(trades["mu"].mean())),
    }


def equity_metrics(curve: pd.DataFrame, trades: pd.DataFrame, initial: float, period_ns: int) -> dict:
    eq = curve["equity"].to_numpy(float)
    dd_frac, dd_abs = max_drawdown(eq)
    out = {"initial_equity": initial, "final_equity": round(float(eq[-1]), 6),
           "total_return": round(float(eq[-1] / initial - 1.0), 6), "max_drawdown": round(dd_frac, 6),
           "max_drawdown_usd": round(dd_abs, 6)}
    if not trades.empty:
        per = trades.groupby(trades["t_exit"] // period_ns)["pnl"].sum()
        out["period_pnl_es_95_usd"] = round(expected_shortfall(per.to_numpy(), 0.05), 6)
        # worst intratrade equity using adverse excursion
        worst = (trades["equity_before"] - trades["notional"] * trades["mae"].abs().fillna(0)).min()
        out["worst_intratrade_equity"] = round(float(worst), 6)
    return out


def period_series(trades: pd.DataFrame, start: int, end: int, period_ns: int) -> pd.Series:
    """Unit-notional net return summed per period, zero-filled (for
    cross-model comparison independent of sizing)."""
    idx = np.arange(start // period_ns, (end - 1) // period_ns + 1)
    s = pd.Series(0.0, index=idx)
    if not trades.empty:
        g = trades.groupby(trades["t_entry"] // period_ns)["ret"].sum()
        s.loc[g.index.intersection(idx)] += g[g.index.intersection(idx)]
    return s


def regime_breakdown(trades: pd.DataFrame, regime_cols: list[str], block_ns: int, conf: float) -> dict:
    """Per-regime net return; regime columns must already be on ``trades``."""
    if trades.empty:
        return {}
    t = trades
    out = {}
    for col in regime_cols:
        out[col] = {}
        for k, g in t.groupby(col):
            m, se, _ = cluster_mean_se(g["ret"].to_numpy(float), time_clusters(g["t_entry"].to_numpy(), block_ns))
            z = 1.96 if conf >= 0.95 else 1.645
            out[col][str(k)] = {"n": int(len(g)), "mean_net_bps": _bps(m),
                                "ci_bps": [_bps(m - z * se), _bps(m + z * se)] if np.isfinite(se) else None}
    return out


def rejection_summary(rej: Counter, n_signals: int, n_trades: int) -> dict:
    return {"signals": n_signals, "executed": n_trades, "rejected": int(sum(rej.values())),
            "reasons": dict(sorted(rej.items(), key=lambda kv: -kv[1]))}

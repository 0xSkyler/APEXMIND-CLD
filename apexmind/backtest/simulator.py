"""Execution-aware portfolio simulation over out-of-sample decisions.

Per-signal executable outcomes (entry after sampled latency, book walks,
fees, funding, passive queue fills) come from the label engine. This layer
adds what a single-trade label cannot know: capital, concurrent positions,
exchange minimum sizes, margin, drawdown-scaled risk and the choice between
aggressive and passive execution. Every signal that does not become a trade
is recorded with a reason.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apexmind.config import Config
from apexmind.execution.optimizer import ExecContext, choose_execution
from apexmind.portfolio.allocator import Allocator, Candidate, OpenPosition
from apexmind.research.policy import Policy
from apexmind.risk.guard import drawdown_multiplier
from apexmind.venues.lighter.markets import LighterMarket


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    rejections: Counter
    equity_curve: pd.DataFrame
    n_signals: int
    initial_equity: float
    final_equity: float
    exec_mode: str
    notes: list[str] = field(default_factory=list)


def simulate(df: pd.DataFrame, decisions: pd.DataFrame, policy: Policy, markets: dict[str, LighterMarket],
             cfg: Config, initial_equity: float, exec_mode: str = "aggressive",
             corr=None) -> BacktestResult:
    """``df`` and ``decisions`` are row-aligned and sorted by time."""
    alloc = Allocator(cfg.portfolio, cfg.strategy.lcb_z, cfg.strategy.min_lcb_return, corr)
    sig_idx = np.flatnonzero(decisions["side"].to_numpy() != 0)
    rejections: Counter = Counter()
    equity = high_water = initial_equity
    open_heap: list = []  # (t_close, seq, symbol)
    open_pos: dict[str, dict] = {}
    rows, curve, notes = [], [(int(df["ts_ns"].iloc[0]) if len(df) else 0, equity)], []
    seq = 0
    label_cap = cfg.labels.notional_usd
    cap_bound = 0

    def close_until(t: int) -> None:
        nonlocal equity, high_water
        while open_heap and open_heap[0][0] <= t:
            t_close, _, sym = heapq.heappop(open_heap)
            pos = open_pos.pop(sym)
            equity += pos["pnl"]
            high_water = max(high_water, equity)
            pos["trade"]["equity_after"] = equity
            rows.append(pos["trade"])
            curve.append((t_close, equity))

    side_name = {1: "long", -1: "short"}
    # materialize only the signal rows once (row-wise pandas access is slow)
    sig_rows = df.iloc[sig_idx].to_dict("records")
    sig_dec = decisions.iloc[sig_idx].to_dict("records")
    for r, d in zip(sig_rows, sig_dec):
        t = int(r["ts_ns"])
        close_until(t)
        sym = str(r["symbol"])
        side = int(d["side"])
        name = side_name[side]
        sp = policy.sides[name]
        h = float(d["horizon"])
        if cfg.strategy.one_position_per_market and sym in open_pos:
            rejections["position_open"] += 1
            continue
        if equity <= 0:
            rejections["account_depleted"] += 1
            continue
        mult = drawdown_multiplier(equity, high_water, cfg.risk)
        if mult <= 0:
            rejections["drawdown_halt"] += 1
            continue
        market = markets[sym]
        price = r["lit_ask"] if side > 0 else r["lit_bid"]
        mode = "aggressive"
        if exec_mode == "optimizer":
            choice = choose_execution(ExecContext(
                mu_aggr=float(d["mu"]), se_aggr=float(d["se"]), p_fill=sp.passive_fill,
                mu_pass_filled=sp.passive_mean_if_filled, se_pass=sp.passive_se, z=cfg.strategy.lcb_z,
                allow_passive=cfg.execution.allow_passive, signal_life_s=h, passive_wait_s=cfg.labels.passive_wait_s,
                max_wait_fraction=cfg.execution.passive_max_wait_fraction))
            if choice.mode == "skip":
                rejections[f"execution_{choice.reason}"] += 1
                continue
            mode = choice.mode
        cand = Candidate(sym, side, float(price), float(d["mu"]), float(d["se"]), sp.std, sp.es05)
        held = [OpenPosition(s, p["side"], p["notional"], p["leverage"], p["std"]) for s, p in open_pos.items()]
        # labels were computed at label_cap notional; larger orders would walk
        # deeper than measured, so sizing never extrapolates beyond it.
        size = alloc.size(equity, cand, market, held, mult, max_notional=label_cap)
        if not size.ok:
            rejections[size.reason] += 1
            continue
        notional, qty = size.notional, size.qty
        if notional >= label_cap * 0.98:
            cap_bound += 1
        if mode == "passive":
            pf = r.get(f"pfill_{name}")
            if not (pf == pf):
                rejections["execution_unavailable"] += 1
                continue
            if pf != 1.0:
                rejections["passive_unfilled"] += 1
                continue
            ret = r[f"pret_{name}_{h:g}"]
            t_entry = r["t_entry_ns"] + r[f"pwait_ms_{name}"] * 1e6
            fee_legs = (market.maker_fee + market.taker_fee)
            gross = math.nan
            fund = r[f"fund_{name}_{h:g}"]
        else:
            ret = r[f"ret_{name}_{h:g}"]
            t_entry = r["t_entry_ns"]
            gross = r[f"gross_{name}_{h:g}"]
            fund = r[f"fund_{name}_{h:g}"]
            fee_legs = 2 * market.taker_fee
        if not (ret == ret):
            rejections["execution_unavailable"] += 1
            continue
        hold_ms = r[f"hold_ms_{h:g}"]
        t_close = int(t_entry + (hold_ms if hold_ms == hold_ms else h * 1e3) * 1e6)
        mae = r.get(f"mae_{name}_{h:g}", math.nan)
        pnl = notional * ret
        # cross-margin liquidation check at the worst executable mark
        maint = notional * market.mmf + sum(p["notional"] * markets[s].mmf for s, p in open_pos.items())
        if mae == mae and equity - notional * abs(mae) < maint:
            pnl = -min(equity, notional * abs(mae) + notional * market.taker_fee)
            notes.append(f"liquidation at {t} {sym}")
        trade = {
            "symbol": sym, "side": side, "t_decision": t, "t_entry": int(t_entry), "t_exit": t_close, "horizon_s": h,
            "exec_mode": mode, "notional": notional, "qty": qty, "leverage": size.leverage, "size_reason": size.reason,
            "ret": float(ret), "gross": float(gross) if gross == gross else math.nan,
            "fees": float(fee_legs), "funding": float(fund) if fund == fund else 0.0,
            "slip_bps": float(r[f"slip_bps_{name}"]), "mae": float(mae), "lat_entry_ms": float(r["lat_entry_ms"]),
            "mu": float(d["mu"]), "lcb": float(d["mu"] - cfg.strategy.lcb_z * d["se"]), "pnl": float(pnl),
            "equity_before": equity, "risk_mult": mult,
            "ref_rv_bps": float(r["ref_rv_bps"]), "lit_spread_bps": float(r["lit_spread_bps"]),
            "trend_60": float(r.get("trend_60", math.nan)),
        }
        seq += 1
        open_pos[sym] = {"side": side, "notional": notional, "leverage": size.leverage, "std": sp.std, "pnl": pnl,
                         "trade": trade}
        heapq.heappush(open_heap, (t_close, seq, sym))
    close_until(2**63 - 1)
    if cap_bound:
        notes.append(f"{cap_bound} orders capped at the label notional ${label_cap:,.0f}")
    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values("t_exit").reset_index(drop=True)
    eq = pd.DataFrame(curve, columns=["ts_ns", "equity"])
    return BacktestResult(trades, rejections, eq, int(len(sig_idx)), initial_equity, equity, exec_mode, notes)

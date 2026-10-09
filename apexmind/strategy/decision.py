"""Per-opportunity decision pipeline (directive section 5).

Steps, each of which can reject with a recorded reason:

1. instrument eligibility (mapped, active, strategy not suspended);
2. synchronized market data (usable books, fresh feeds, cross-venue skew,
   clock uncertainty);
3. cross-venue price-discovery signal (model score per side/horizon);
4. continuation/reversal probability (calibrated on held-out data);
5. executable return distribution (calibrated mean, se, std, ES, MAE);
6. live cost adjustments (spread/impact drift vs calibration, funding due
   inside the holding period);
7. uncertainty-aware expected value (LCB must exceed the margin);
8. interaction with existing positions (one per market, covariance);
9. feasible notional and leverage (allocator; exchange minimums);
10. execution choice; submit only if the expected contribution is positive.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from apexmind.config import Config
from apexmind.execution.optimizer import ExecContext, choose_execution
from apexmind.portfolio.allocator import Allocator, Candidate, OpenPosition
from apexmind.research.policy import Policy
from apexmind.risk.guard import check_data
from apexmind.venues.lighter.markets import LighterMarket


@dataclass
class StrategyBundle:
    """Everything needed to trade a validated model (stored by the lab)."""

    name: str
    model: object
    policy: Policy
    feature_names: list[str]
    calib_spread_bps: dict[str, float]
    calib_impact_bps: dict[str, float]
    calib_latency_ms: float
    manifest: dict = field(default_factory=dict)


@dataclass
class Decision:
    t_ns: int
    symbol: str
    step: int  # last step reached (1..10); 10 + ok means submit
    ok: bool
    reason: str
    side: int = 0
    horizon_s: float = math.nan
    p_favorable: float = math.nan
    mu: float = math.nan
    se: float = math.nan
    lcb: float = math.nan
    funding_cost: float = 0.0
    notional: float = 0.0
    qty: float = 0.0
    leverage: float = 0.0
    exec_mode: str = ""
    price: float = math.nan
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionEngine:
    def __init__(self, cfg: Config, bundle: StrategyBundle, markets: dict[str, LighterMarket],
                 eligible: set[str], corr=None) -> None:
        self.cfg, self.bundle, self.markets, self.eligible = cfg, bundle, markets, eligible
        self.alloc = Allocator(cfg.portfolio, cfg.strategy.lcb_z, cfg.strategy.min_lcb_return, corr)
        self.suspended: set[str] = set()
        self._lags: tuple[int, ...] = tuple(getattr(bundle.model, "LAGS", ())) if getattr(
            bundle.model, "justified", False) else ()
        self._hist: dict[str, deque] = {}

    def observe(self, features_by_symbol: dict[str, list[float]]) -> None:
        """Called on every grid tick for every symbol: keeps the history that
        lag-feature (temporal) models need, identical to research."""
        if self._lags:
            for s, v in features_by_symbol.items():
                self._hist.setdefault(s, deque(maxlen=max(self._lags) + 1)).append(v)

    def _frame(self, syms: list[str], features_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
        names = self.bundle.feature_names
        frame = pd.DataFrame([features_by_symbol[s] for s in syms], columns=names)
        if not self._lags:
            return frame
        extra = {}
        for k in self._lags:
            rows = []
            for s in syms:
                h = self._hist.get(s, ())
                rows.append(h[-1 - k] if len(h) > k else [math.nan] * len(names))
            lagged = np.asarray(rows, float)
            for j, n in enumerate(names):
                extra[f"{n}__lag{k}"] = lagged[:, j]
        return pd.concat([frame, pd.DataFrame(extra)], axis=1)

    def policy_rows(self, features_by_symbol: dict[str, list[float]]) -> dict[str, dict]:
        """Score several symbols in one model call (steps 3-5)."""
        syms = list(features_by_symbol)
        frame = self._frame(syms, features_by_symbol)
        scores = {k: np.asarray(v, float) for k, v in self.bundle.model.score(frame).items()}
        dec = self.bundle.policy.decide(scores, len(syms))
        return {s: dec.iloc[i].to_dict() for i, s in enumerate(syms)}

    def decide(self, t_ns: int, symbol: str, features: list[float], info: dict, *, equity: float,
               open_positions: list[OpenPosition], risk_multiplier: float, lit_staleness_ms: float,
               ref_staleness_ms: float, clock_uncertainty_ms: float, latency_ms_now: float,
               tx_capacity: float, funding_rate: float, funding_eta_s: float, spread_bps: float,
               impact_bps: float, policy_row: dict | None = None) -> Decision:
        cfg = self.cfg
        D = lambda step, reason, **kw: Decision(t_ns, symbol, step, False, reason, **kw)  # noqa: E731
        # 1. eligibility
        m = self.markets.get(symbol)
        if symbol not in self.eligible or m is None or not m.active:
            return D(1, "instrument_ineligible")
        if symbol in self.suspended or self.bundle.name in self.suspended:
            return D(1, "strategy_suspended")
        # 2. synchronized data
        dc = check_data(lit_staleness_ms, ref_staleness_ms, clock_uncertainty_ms, bool(info.get("valid")), cfg.risk)
        if not dc.ok:
            return D(2, dc.reason)
        if abs(lit_staleness_ms - ref_staleness_ms) > cfg.strategy.max_data_skew_ms:
            return D(2, "cross_venue_skew")
        # 3-5. signal and calibrated distribution
        dec = policy_row if policy_row is not None else self.policy_rows({symbol: features})[symbol]
        side = int(dec["side"])
        if side == 0:
            return D(3, "no_signal")
        name = "long" if side > 0 else "short"
        sp = self.bundle.policy.sides[name]
        mu, se, h = float(dec["mu"]), float(dec["se"]), float(dec["horizon"])
        # 6. live cost adjustments
        funding_cost = 0.0
        hold_s = h + 2 * latency_ms_now / 1e3 if latency_ms_now == latency_ms_now else h
        if funding_rate == funding_rate and 0 <= funding_eta_s <= hold_s:
            funding_cost = funding_rate * side  # paid by longs when positive
        mu_adj = mu - funding_cost
        lcb = mu_adj - cfg.strategy.lcb_z * se
        common = dict(side=side, horizon_s=h, p_favorable=sp.p_pos, mu=mu_adj, se=se, lcb=lcb, funding_cost=funding_cost)
        # 7. uncertainty-aware EV
        if not (lcb > cfg.strategy.min_lcb_return):
            return D(7, "lcb_below_margin", **common)
        # 8. existing positions
        if cfg.strategy.one_position_per_market and any(p.symbol == symbol for p in open_positions):
            return D(8, "position_open", **common)
        # 9. size and leverage
        price = info["lit_ask"] if side > 0 else info["lit_bid"]
        cand = Candidate(symbol, side, float(price), mu_adj, se, sp.std, sp.es05)
        size = self.alloc.size(equity, cand, m, open_positions, risk_multiplier,
                               max_notional=self.cfg.labels.notional_usd)
        if not size.ok:
            return D(9, size.reason, price=price, **common)
        # 10. execution
        choice = choose_execution(ExecContext(
            mu_aggr=mu_adj, se_aggr=se, p_fill=sp.passive_fill, mu_pass_filled=sp.passive_mean_if_filled - funding_cost,
            se_pass=sp.passive_se, z=cfg.strategy.lcb_z, allow_passive=cfg.execution.allow_passive,
            signal_life_s=h, passive_wait_s=cfg.labels.passive_wait_s,
            spread_bps_now=spread_bps, spread_bps_cal=self.bundle.calib_spread_bps.get(symbol, math.nan),
            impact_bps_now=impact_bps, impact_bps_cal=self.bundle.calib_impact_bps.get(symbol, math.nan),
            latency_ms_now=latency_ms_now, latency_ms_cal=self.bundle.calib_latency_ms, tx_capacity=tx_capacity,
            max_wait_fraction=cfg.execution.passive_max_wait_fraction))
        if choice.mode == "skip":
            return D(10, f"execution_{choice.reason}", price=price, **common)
        return Decision(t_ns, symbol, 10, True, size.reason, notional=size.notional, qty=size.qty,
                        leverage=size.leverage, exec_mode=choice.mode, price=float(price),
                        extra={"exec_ev": choice.ev, "exec_lcb": choice.ev_lcb}, **common)

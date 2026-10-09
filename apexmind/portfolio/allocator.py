"""Capital allocation: fractional-Kelly growth sizing under hard constraints.

For a candidate with calibrated expected net return ``mu`` (we use the
lower confidence bound) and return std ``sigma``, the log-growth-optimal
fraction of equity committed as notional is ``(mu - cov) / sigma**2``,
where ``cov`` is the covariance with positions already held. We take a
fraction of that (``kelly_fraction``) and then apply hard caps:

* per-position notional / equity;
* gross notional / equity;
* initial margin usage (at the configured exchange leverage) / equity;
* a per-trade loss budget using the calibrated 5% expected shortfall.

Exchange minimums: if the sized order is below the venue minimum, the
minimum order is used only if it still satisfies every hard cap *and* still
has positive expected log-growth; otherwise the trade is rejected. Leverage
is never raised to make a minimum order fit. Nothing here assumes a minimum
account size: a $10 account trades whenever the venue minimum allows it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from apexmind.config import PortfolioConfig
from apexmind.venues.lighter.markets import LighterMarket


@dataclass
class Candidate:
    symbol: str
    side: int  # +1 long, -1 short
    price: float
    mu: float  # calibrated expected net return per unit notional
    se: float  # standard error of mu
    std: float  # per-trade return std
    es05: float  # 5% expected shortfall of per-trade returns (negative)


@dataclass
class OpenPosition:
    symbol: str
    side: int
    notional: float
    leverage: float
    std: float


@dataclass
class SizeDecision:
    notional: float
    qty: float
    leverage: float
    fraction: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.qty > 0


def _reject(reason: str) -> SizeDecision:
    return SizeDecision(0.0, 0.0, 0.0, 0.0, reason)


class Allocator:
    def __init__(self, cfg: PortfolioConfig, lcb_z: float, min_lcb: float = 0.0,
                 corr: Callable[[str, str], float] | None = None) -> None:
        self.cfg, self.z, self.min_lcb = cfg, lcb_z, min_lcb
        self.corr = corr or (lambda a, b: 1.0 if a == b else 0.5)

    def leverage_for(self, market: LighterMarket) -> float:
        """Exchange leverage setting: the configured gross cap, bounded by the
        market maximum. It is a property of the account configuration, not
        something adjusted per trade."""
        return max(1.0, min(market.max_leverage, math.ceil(self.cfg.max_gross_leverage)))

    def size(self, equity: float, cand: Candidate, market: LighterMarket, open_positions: list[OpenPosition],
             risk_multiplier: float = 1.0, max_notional: float = math.inf) -> SizeDecision:
        """``max_notional``: the largest order size at which execution costs
        were validated (the label notional); never extrapolate beyond it."""
        cfg = self.cfg
        E = equity - cfg.reserve_usd
        if E <= 0:
            return _reject("no_free_equity")
        if risk_multiplier <= 0:
            return _reject("risk_halt")
        mu_lcb = cand.mu - self.z * cand.se
        if not (mu_lcb > self.min_lcb):
            return _reject("ev_lcb_not_positive")
        if not (cand.std > 0) or not (cand.price > 0):
            return _reject("invalid_risk_estimate")
        var = cand.std**2
        cov = 0.0
        for p in open_positions:
            w = p.side * p.notional / E
            cov += w * self.corr(cand.symbol, p.symbol) * cand.std * p.std * cand.side
        f_opt = (mu_lcb - cov) / var
        if f_opt <= 0:
            return _reject("portfolio_correlation")
        lev = self.leverage_for(market)
        gross_used = sum(p.notional for p in open_positions) / E
        margin_used = sum(p.notional / p.leverage for p in open_positions) / E
        caps = {
            "position": cfg.max_position_equity_fraction,
            "gross": cfg.max_gross_leverage - gross_used,
            "margin": (cfg.max_margin_utilization - margin_used) * lev,
            "loss_budget": cfg.max_trade_loss_equity_fraction / abs(cand.es05) if cand.es05 < 0 else math.inf,
            "validated_notional": max_notional / E,
        }
        hard_cap = min(caps.values())
        if hard_cap <= 0:
            binding = min(caps, key=caps.get)
            return _reject(f"cap_{binding}_exhausted")
        f = min(cfg.kelly_fraction * f_opt * risk_multiplier, hard_cap)
        qty = market.quantize_size(f * E / cand.price)
        reason = "ok"
        if not market.order_is_feasible(qty, cand.price):
            q_min = market.min_feasible_qty(cand.price)
            f_min = q_min * cand.price / E
            growth = f_min * mu_lcb - 0.5 * f_min**2 * var - f_min * cov
            if f_min <= hard_cap and growth > 0:
                qty, reason = q_min, "raised_to_exchange_minimum"
            else:
                why = "cap" if f_min > hard_cap else "negative_growth"
                return _reject(f"below_exchange_minimum_{why}")
        notional = qty * cand.price
        return SizeDecision(notional, qty, lev, notional / E, reason)

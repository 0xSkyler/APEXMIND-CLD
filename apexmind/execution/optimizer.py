"""Exchange-native execution choice: aggressive (IOC/market) or passive
(post-only at the touch), or skip.

Both alternatives are valued with what was *measured* on calibration data
for this signal: the aggressive expectation already includes latency drift,
spread, impact and taker fees; the passive alternative is worth
``P(fill) * E[return | filled]`` where the conditional expectation embeds the
adverse selection of getting filled. Current conditions adjust the
aggressive value for spread and impact that differ from calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExecContext:
    mu_aggr: float  # calibrated expected net return, aggressive entry
    se_aggr: float
    p_fill: float  # passive fill probability within the wait window
    mu_pass_filled: float  # expected net return given a passive fill
    se_pass: float
    z: float
    allow_passive: bool
    signal_life_s: float  # remaining useful life (the policy horizon)
    passive_wait_s: float
    spread_bps_now: float = math.nan
    spread_bps_cal: float = math.nan
    impact_bps_now: float = math.nan
    impact_bps_cal: float = math.nan
    latency_ms_now: float = math.nan
    latency_ms_cal: float = math.nan
    tx_capacity: float = math.inf  # remaining transactions we may send
    max_wait_fraction: float = 0.5


@dataclass
class ExecChoice:
    mode: str  # "aggressive" | "passive" | "skip"
    ev: float
    ev_lcb: float
    reason: str


def _nz(x: float) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else x


def choose_execution(c: ExecContext) -> ExecChoice:
    if c.tx_capacity < 2:  # need at least an entry and an exit
        return ExecChoice("skip", 0.0, 0.0, "transaction_capacity")
    if not math.isnan(c.latency_ms_now) and not math.isnan(c.latency_ms_cal) and c.latency_ms_now > 2 * c.latency_ms_cal:
        return ExecChoice("skip", 0.0, 0.0, "latency_degraded")
    # cost drift vs calibration conditions (bps -> fraction); half spread paid on entry
    drift = 0.5 * _nz(c.spread_bps_now - c.spread_bps_cal) + _nz(c.impact_bps_now - c.impact_bps_cal)
    aggr = c.mu_aggr - max(drift, 0.0) * 1e-4
    aggr_lcb = aggr - c.z * c.se_aggr
    best = ExecChoice("aggressive", aggr, aggr_lcb, "aggressive_best")
    passive_ok = (c.allow_passive and not math.isnan(c.p_fill) and not math.isnan(c.mu_pass_filled)
                  and c.passive_wait_s <= c.max_wait_fraction * c.signal_life_s)
    if passive_ok:
        ev = c.p_fill * c.mu_pass_filled
        lcb = c.p_fill * (c.mu_pass_filled - c.z * _nz(c.se_pass))
        if lcb > best.ev_lcb:
            best = ExecChoice("passive", ev, lcb, "passive_best")
    if not (best.ev_lcb > 0):
        return ExecChoice("skip", best.ev, best.ev_lcb, "no_positive_execution")
    return best

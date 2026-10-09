"""Risk controls shared by backtest and live trading.

* Drawdown scaling: risk shrinks linearly from ``drawdown_scale_start`` to
  zero at ``max_drawdown_halt`` (of high-water equity).
* Pre-trade data checks: stale feeds or excessive clock uncertainty block
  new entries.
* Margin protection: maintenance requirement relative to equity drives
  warn / reduce-only states.
* Live edge monitor: compares realized net returns to the calibrated
  expectation; persistent shortfall halves the strategy's size multiplier and
  eventually suspends it. The multiplier never exceeds 1, so a strategy can
  only be scaled back up to its validated size - never levered beyond it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from apexmind.config import RiskConfig


def drawdown_multiplier(equity: float, high_water: float, cfg: RiskConfig) -> float:
    if high_water <= 0:
        return 0.0
    dd = max(0.0, 1.0 - equity / high_water)
    if dd >= cfg.max_drawdown_halt:
        return 0.0
    if dd <= cfg.drawdown_scale_start:
        return 1.0
    return 1.0 - (dd - cfg.drawdown_scale_start) / (cfg.max_drawdown_halt - cfg.drawdown_scale_start)


@dataclass
class DataCheck:
    ok: bool
    reason: str = ""


def check_data(lit_staleness_ms: float, ref_staleness_ms: float, clock_uncertainty_ms: float, books_usable: bool,
               cfg: RiskConfig) -> DataCheck:
    if not books_usable:
        return DataCheck(False, "book_unusable")
    if not (lit_staleness_ms <= cfg.max_feed_staleness_ms):
        return DataCheck(False, "lighter_feed_stale")
    if not (ref_staleness_ms <= cfg.max_feed_staleness_ms):
        return DataCheck(False, "reference_feed_stale")
    if not (clock_uncertainty_ms <= cfg.max_clock_uncertainty_ms):
        return DataCheck(False, "clock_uncertain")
    return DataCheck(True)


def margin_state(maintenance_requirement: float, equity: float, cfg: RiskConfig) -> str:
    """'ok' | 'warn' (no new risk) | 'reduce' (reduce-only, shrink positions)."""
    if equity <= 0:
        return "reduce"
    ratio = maintenance_requirement / equity
    if ratio >= cfg.margin_reduce_ratio:
        return "reduce"
    if ratio >= cfg.margin_warning_ratio:
        return "warn"
    return "ok"


@dataclass
class EdgeMonitor:
    """Sequential check of live performance against calibration.

    After each closed trade, with at least ``min_trades`` in the window:
    if the upper 95% bound of realized mean net return is below zero, or the
    realized mean is below the calibrated LCB by more than 2 standard errors,
    the size multiplier is halved. After ``suspend_after`` reductions the
    strategy is suspended. The multiplier recovers (to at most 1.0) only when
    the realized lower bound is back above zero.
    """

    cfg: RiskConfig
    expected_lcb: float
    window: deque = field(default_factory=deque)
    multiplier: float = 1.0
    reductions: int = 0
    suspended: bool = False
    last_action: str = "none"
    _since_action: int = 0

    def record(self, net_return: float) -> str:
        self.window.append(net_return)
        while len(self.window) > self.cfg.edge_window_trades:
            self.window.popleft()
        self._since_action += 1
        n = len(self.window)
        if self.suspended or n < self.cfg.edge_min_trades or self._since_action < self.cfg.edge_min_trades // 2:
            return "hold"
        x = np.asarray(self.window)
        m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
        if m + 1.96 * se < 0 or m < self.expected_lcb - 2 * se:
            self.reductions += 1
            self._since_action = 0
            if self.reductions >= self.cfg.edge_suspend_after_reductions:
                self.suspended, self.multiplier, self.last_action = True, 0.0, "suspend"
                return "suspend"
            self.multiplier *= 0.5
            self.last_action = "reduce"
            return "reduce"
        if m - 1.96 * se > 0 and self.multiplier < 1.0:
            self._since_action = 0
            self.multiplier = min(1.0, self.multiplier * 2)
            self.last_action = "restore"
            return "restore"
        return "hold"

    def state(self) -> dict:
        return {"multiplier": self.multiplier, "reductions": self.reductions, "suspended": self.suspended,
                "n": len(self.window), "last_action": self.last_action, "expected_lcb": self.expected_lcb,
                "window": list(self.window)}

    @classmethod
    def restore(cls, cfg: RiskConfig, d: dict) -> "EdgeMonitor":
        m = cls(cfg, d["expected_lcb"])
        m.window = deque(d.get("window", []))
        m.multiplier, m.reductions = d["multiplier"], d["reductions"]
        m.suspended, m.last_action = d["suspended"], d.get("last_action", "none")
        return m

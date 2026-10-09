import math

import numpy as np

from apexmind.config import Config
from apexmind.portfolio.allocator import OpenPosition
from apexmind.research.policy import Policy, SidePolicy
from apexmind.strategy.decision import DecisionEngine, StrategyBundle
from apexmind.venues.lighter.markets import LighterMarket


class StubModel:
    def __init__(self, score):
        self.s = score

    def score(self, df):
        n = len(df)
        return {("long", 15.0): np.full(n, self.s), ("short", 15.0): np.full(n, -self.s)}


def side(lcb=0.0004, mean=0.0006, se=0.0001, **kw):
    base = dict(side="long", horizon=15.0, threshold=1.0, n_cal=500, mean=mean, se=se, lcb=lcb, std=0.002,
                p_pos=0.6, q05=-0.003, q95=0.004, es05=-0.004, mae_mean=-0.001, p_sig_slip=0.05, hold_ms=15500.0)
    base.update(kw)
    return SidePolicy(**base)


def engine(score=2.0, sp=None, **cfg_kw):
    cfg = Config()
    for k, v in cfg_kw.items():
        setattr(cfg.strategy, k, v)
    pol = Policy("stub", {"long": sp or side(), "short": None}, {})
    b = StrategyBundle("stub", StubModel(score), pol, ["f"], {"ETH": 1.0}, {"ETH": 0.5}, 300.0)
    m = LighterMarket(0, "ETH", "active", 0.0002, 0.00002, 0.005, 10.0, 4, 2, 0.05, 0.02, 0.012)
    return DecisionEngine(cfg, b, {"ETH": m}, {"ETH"})


INFO = {"valid": True, "lit_bid": 2999.0, "lit_ask": 3000.0}


def run(e, sym="ETH", info=INFO, **kw):
    args = dict(equity=1000.0, open_positions=[], risk_multiplier=1.0, lit_staleness_ms=50.0,
                ref_staleness_ms=40.0, clock_uncertainty_ms=5.0, latency_ms_now=300.0, tx_capacity=math.inf,
                funding_rate=0.0, funding_eta_s=1800.0, spread_bps=1.0, impact_bps=0.5)
    args.update(kw)
    return e.decide(1, sym, [0.0], info, **args)


def test_happy_path_submits():
    d = run(engine())
    assert d.ok and d.step == 10 and d.side == 1 and d.exec_mode == "aggressive" and d.qty > 0
    assert d.lcb > 0 and d.horizon_s == 15.0


def test_each_step_rejects_with_reason():
    assert run(engine(), sym="BTC").reason == "instrument_ineligible"
    e = engine()
    e.suspended.add("stub")
    assert run(e).reason == "strategy_suspended"
    assert run(engine(), info={**INFO, "valid": False}).reason == "book_unusable"
    assert run(engine(), lit_staleness_ms=10_000).reason == "lighter_feed_stale"
    assert run(engine(), lit_staleness_ms=2900, ref_staleness_ms=10).reason == "cross_venue_skew"
    assert run(engine(score=0.5)).reason == "no_signal"
    d = run(engine(sp=side(se=0.001)))
    assert d.step == 7 and d.reason == "lcb_below_margin"
    held = [OpenPosition("ETH", 1, 100.0, 4.0, 0.002)]
    assert run(engine(), open_positions=held).reason == "position_open"
    assert run(engine(), equity=1.0).reason.startswith("below_exchange_minimum")
    assert run(engine(), tx_capacity=1).reason == "execution_transaction_capacity"


def test_funding_due_inside_holding_period_is_charged():
    # longs pay positive funding; 30 bps due in 5 s wipes out a 6 bps edge
    d = run(engine(), funding_rate=0.003, funding_eta_s=5.0)
    assert not d.ok and d.funding_cost == 0.003 and d.reason == "lcb_below_margin"
    d2 = run(engine(), funding_rate=0.003, funding_eta_s=600.0)
    assert d2.ok and d2.funding_cost == 0.0

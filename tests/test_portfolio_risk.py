import math

import pytest

from apexmind.config import Config, PortfolioConfig, RiskConfig
from apexmind.execution.optimizer import ExecContext, choose_execution
from apexmind.portfolio.allocator import Allocator, Candidate, OpenPosition
from apexmind.risk.guard import EdgeMonitor, check_data, drawdown_multiplier, margin_state
from apexmind.venues.lighter.markets import LighterMarket


def market(min_base=0.005, min_quote=10.0, size_dec=4, min_imf=0.02):
    return LighterMarket(0, "ETH", "active", 0.0002, 0.00002, min_base, min_quote, size_dec, 2, 0.05, min_imf, 0.012)


def cand(mu=0.0008, se=0.0001, std=0.002, es=-0.004, price=3000.0, side=1, sym="ETH"):
    return Candidate(sym, side, price, mu, se, std, es)


def test_kelly_sizing_and_quantization():
    a = Allocator(PortfolioConfig(kelly_fraction=0.25), lcb_z=1.645)
    d = a.size(1000.0, cand(), market(), [])
    mu_lcb = 0.0008 - 1.645 * 0.0001
    f = min(0.25 * mu_lcb / 0.002**2, 2.0, 0.02 / 0.004)
    assert d.ok and d.reason == "ok"
    assert d.notional == pytest.approx(market().quantize_size(f * 1000 / 3000) * 3000)
    assert d.leverage == 4.0  # configured gross cap, not raised per trade


def test_ten_dollar_account_trades_when_minimum_fits():
    a = Allocator(PortfolioConfig(kelly_fraction=0.05), lcb_z=1.645)
    d = a.size(10.0, cand(mu=0.0003, se=0.00005, std=0.004), market(), [])
    # kelly says ~$6.8 but the venue minimum ($15 = 0.005 ETH) fits every cap
    # and still has positive expected log-growth
    assert d.ok and d.reason == "raised_to_exchange_minimum"
    assert d.qty == pytest.approx(0.005)
    assert d.leverage == 4.0


def test_minimum_never_forces_extra_leverage():
    # $5 account: the $15 minimum would need 3x notional/equity > position cap 2x
    a = Allocator(PortfolioConfig(max_position_equity_fraction=2.0), lcb_z=1.645)
    d = a.size(5.0, cand(), market(), [])
    assert not d.ok and d.reason.startswith("below_exchange_minimum")


def test_rejects_without_positive_lcb_or_with_exhausted_caps():
    a = Allocator(PortfolioConfig(), lcb_z=1.645)
    assert a.size(1000, cand(mu=0.0001, se=0.0001), market(), []).reason == "ev_lcb_not_positive"
    held = [OpenPosition("BTC", 1, 4000.0, 4.0, 0.002)]
    assert a.size(1000, cand(), market(), held).reason.startswith("cap_")
    assert a.size(0.0, cand(), market(), []).reason == "no_free_equity"
    assert a.size(1000, cand(), market(), [], risk_multiplier=0.0).reason == "risk_halt"


def test_correlated_same_direction_exposure_reduces_size():
    a = Allocator(PortfolioConfig(kelly_fraction=0.05, max_trade_loss_equity_fraction=1.0), lcb_z=1.645,
                  corr=lambda x, y: 0.9)
    c = cand(mu=0.0003, se=0.00005, std=0.004)  # small enough that no hard cap binds
    alone = a.size(1000, c, market(), [])
    with_pos = a.size(1000, c, market(), [OpenPosition("BTC", 1, 500.0, 4.0, 0.004)])
    hedge = a.size(1000, c, market(), [OpenPosition("BTC", -1, 500.0, 4.0, 0.004)])
    assert with_pos.notional < alone.notional < hedge.notional


def ctx(**kw):
    base = dict(mu_aggr=0.0005, se_aggr=0.0001, p_fill=0.6, mu_pass_filled=0.0006, se_pass=0.0001, z=1.645,
                allow_passive=True, signal_life_s=15.0, passive_wait_s=5.0)
    base.update(kw)
    return ExecContext(**base)


def test_execution_choice():
    assert choose_execution(ctx()).mode == "aggressive"  # 0.6*0.00044 < 0.000336
    assert choose_execution(ctx(p_fill=0.95, mu_pass_filled=0.0009)).mode == "passive"
    assert choose_execution(ctx(allow_passive=False, p_fill=0.95, mu_pass_filled=0.0009)).mode == "aggressive"
    assert choose_execution(ctx(passive_wait_s=10.0, p_fill=0.95, mu_pass_filled=0.0009)).mode == "aggressive"
    # wider spread now than at calibration erodes the aggressive edge to nothing
    assert choose_execution(ctx(allow_passive=False, spread_bps_now=10.0, spread_bps_cal=1.0)).mode == "skip"
    assert choose_execution(ctx(tx_capacity=1)).reason == "transaction_capacity"
    assert choose_execution(ctx(latency_ms_now=900, latency_ms_cal=300)).reason == "latency_degraded"


def test_drawdown_multiplier():
    r = RiskConfig(drawdown_scale_start=0.1, max_drawdown_halt=0.25)
    assert drawdown_multiplier(95, 100, r) == 1.0
    assert drawdown_multiplier(82.5, 100, r) == pytest.approx(0.5)
    assert drawdown_multiplier(75, 100, r) == 0.0


def test_data_and_margin_checks():
    r = RiskConfig()
    assert check_data(10, 10, 5, True, r).ok
    assert check_data(10, 10, 5, False, r).reason == "book_unusable"
    assert check_data(math.inf, 10, 5, True, r).reason == "lighter_feed_stale"
    assert check_data(10, 10, 1000, True, r).reason == "clock_uncertain"
    assert margin_state(10, 100, r) == "ok"
    assert margin_state(60, 100, r) == "warn"
    assert margin_state(80, 100, r) == "reduce"


def test_edge_monitor_reduces_suspends_and_never_levers_up():
    r = RiskConfig(edge_min_trades=20, edge_window_trades=60, edge_suspend_after_reductions=3)
    m = EdgeMonitor(r, expected_lcb=0.0003)
    actions = [m.record(-0.0005 + (0.0001 if i % 2 else -0.0001)) for i in range(200)]
    assert "reduce" in actions and m.suspended and m.multiplier == 0.0
    good = EdgeMonitor(r, expected_lcb=0.0001)
    for i in range(300):
        good.record(0.0004 + (0.0001 if i % 2 else -0.0001))
        assert good.multiplier <= 1.0
    assert not good.suspended and good.multiplier == 1.0
    restored = EdgeMonitor.restore(r, m.state())
    assert restored.suspended and restored.multiplier == 0.0


def test_defaults_have_no_minimum_deposit():
    cfg = Config()
    assert cfg.portfolio.reserve_usd == 0.0
    assert 10.0 in cfg.research.initial_equities

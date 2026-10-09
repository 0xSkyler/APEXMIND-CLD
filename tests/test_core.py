import math
import os
import stat

import pytest

from apexmind.config import SecretError, load_config, load_lighter_api_key
from apexmind.core.clock import FeedDelayMonitor, ManualClock, OffsetEstimator, RollingQuantiles
from apexmind.core.events import BUY, SELL
from apexmind.core.orderbook import L2Book
from apexmind.core.ratelimit import TokenBucket


def make_book():
    b = L2Book()
    b.apply_snapshot(bids=[(99.0, 2.0), (98.0, 5.0), (97.0, 10.0)], asks=[(101.0, 1.0), (102.0, 4.0), (103.0, 10.0)])
    return b


def test_book_top_and_mid():
    b = make_book()
    assert b.best_bid() == (99.0, 2.0)
    assert b.best_ask() == (101.0, 1.0)
    assert b.mid() == 100.0
    assert b.spread() == 2.0
    assert b.top(BUY, 2) == [(99.0, 2.0), (98.0, 5.0)]
    assert b.top(SELL, 2) == [(101.0, 1.0), (102.0, 4.0)]
    # microprice leans toward the thinner side
    assert b.microprice() == pytest.approx((99 * 1 + 101 * 2) / 3)


def test_book_delta_and_removal():
    b = make_book()
    b.apply_delta(bids=[(99.0, 0.0), (99.5, 1.0)], asks=[(101.0, 0.0)])
    assert b.best_bid() == (99.5, 1.0)
    assert b.best_ask() == (102.0, 4.0)


def test_walk_qty_and_notional():
    b = make_book()
    f = b.walk(BUY, qty=3.0)
    assert f.complete
    assert f.vwap == pytest.approx((101 * 1 + 102 * 2) / 3)
    assert f.worst_price == 102.0
    f = b.walk(SELL, notional=99 * 2 + 98 * 1)
    assert f.complete
    assert f.qty == pytest.approx(3.0)
    f = b.walk(BUY, qty=100.0)
    assert not f.complete
    assert f.qty == pytest.approx(15.0)


def test_impact_bps_sign():
    b = make_book()
    assert b.impact_bps(BUY, 101.0) == pytest.approx(100.0)  # 1% above mid
    assert b.impact_bps(SELL, 99.0) == pytest.approx(100.0)


def test_set_bbo_removes_stale_levels():
    b = make_book()
    b.set_bbo(98.0, 3.0, 102.0, 1.5)
    assert b.best_bid() == (98.0, 3.0)
    assert b.best_ask() == (102.0, 1.5)
    assert 99.0 not in b.bids and 101.0 not in b.asks


def test_crossed_book_not_usable():
    b = make_book()
    b.apply_delta(bids=[(102.0, 1.0)], asks=[])
    assert b.is_crossed() and not b.usable()


def test_depth_within_band():
    b = make_book()
    assert b.depth_notional_within(BUY, 150) == pytest.approx(99 * 2)
    assert b.depth_notional_within(BUY, 250) == pytest.approx(99 * 2 + 98 * 5)
    assert b.depth_notional_within(SELL, 250) == pytest.approx(101 * 1 + 102 * 4)


def test_offset_estimator_prefers_min_rtt():
    e = OffsetEstimator()
    # true offset +5ms; noisy probe with large rtt reports a biased offset
    e.add_probe(0, 5_000_000 + 40_000_000, 100_000_000)
    e.add_probe(1_000_000_000, 1_000_000_000 + 5_000_000 + 1_000_000, 1_000_000_000 + 2_000_000)
    assert e.offset_ns == pytest.approx(5_000_000)
    assert e.uncertainty_ns == pytest.approx(1_000_000)


def test_feed_delay_monitor():
    m = FeedDelayMonitor()
    for i in range(100):
        m.observe("x", ts_local_ns=i * 1_000_000_000 + 20_000_000, ts_exch_ns=i * 1_000_000_000)
    assert m.delay_ms("x") == pytest.approx(20.0)
    assert m.staleness_ms("x", 99 * 1_000_000_000 + 520_000_000) == pytest.approx(500.0)
    assert math.isinf(m.staleness_ms("missing", 0))


def test_rolling_quantiles_window():
    rq = RollingQuantiles(window=5)
    for x in range(10):
        rq.add(float(x))
    assert len(rq) == 5
    assert rq.quantile(0.0) == 5.0 and rq.quantile(1.0) == 9.0


def test_manual_clock_monotone():
    c = ManualClock(10)
    c.advance(5)
    assert c.now_ns() == 15
    with pytest.raises(ValueError):
        c.set(1)


def test_token_bucket():
    t = [0.0]
    tb = TokenBucket(2.0, burst=2.0, clock=lambda: t[0])
    assert tb.try_acquire() and tb.try_acquire()
    assert not tb.try_acquire()
    t[0] += 0.5
    assert tb.try_acquire()
    assert tb.wait_time() == pytest.approx(0.5)


def test_config_env_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("features:\n  grid_ms: 500\nlabels:\n  horizons_s: [1, 2]\n")
    cfg = load_config(p, env={"APEXMIND__PORTFOLIO__KELLY_FRACTION": "0.1"})
    assert cfg.features.grid_ms == 500
    assert cfg.labels.horizons_s == [1.0, 2.0]
    assert cfg.portfolio.kelly_fraction == 0.1


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("features:\n  grdi_ms: 500\n")
    with pytest.raises(ValueError):
        load_config(p, env={})


def test_secret_file_permissions(tmp_path):
    cfg = load_config(None, env={}).live
    key = tmp_path / "k"
    key.write_text("abc\n")
    os.chmod(key, 0o644)
    cfg.secrets_file = str(key)
    with pytest.raises(SecretError):
        load_lighter_api_key(cfg, env={})
    os.chmod(key, stat.S_IRUSR | stat.S_IWUSR)
    assert load_lighter_api_key(cfg, env={}) == "abc"
    assert load_lighter_api_key(cfg, env={"APEXMIND_LIGHTER_API_KEY": "zz"}) == "zz"

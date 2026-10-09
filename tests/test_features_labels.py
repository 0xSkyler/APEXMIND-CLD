
import numpy as np
import pytest

from apexmind.config import Config
from apexmind.core.events import BBO, LIGHTER, SELL, BookDelta, BookSnapshot, MarketStats, Trade
from apexmind.data.dataset import assemble_frame, build_dataset, run_pipeline
from apexmind.data.synthetic import SymbolSpec, SyntheticSpec, generate
from apexmind.latency.model import LatencyModel

REF = "binance_usdm"
S = 10**9


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    truth = generate(SyntheticSpec(hours=0.25, symbols=[SymbolSpec()], seed=3), str(d))
    return str(d), truth["spec"]["start_ns"]


def small_cfg():
    cfg = Config()
    cfg.labels.latency_source = "prior"
    cfg.labels.horizons_s = [5.0]
    return cfg


def test_no_lookahead_features(synth):
    """Features at t are identical whether or not data after t exists."""
    d, s0 = synth
    cfg = small_cfg()
    start = s0 + 4 * 60 * S
    full, _ = build_dataset(cfg, d, start, s0 + 14 * 60 * S)
    cut, _ = build_dataset(cfg, d, start, s0 + 9 * 60 * S)
    names = full.attrs["feature_names"]
    a = full[full.ts_ns.isin(cut.ts_ns)].reset_index(drop=True)
    assert len(a) == len(cut) > 1000
    np.testing.assert_array_equal(a[names].to_numpy(), cut[names].to_numpy())
    # labels: rows whose simulated orders finished before the cut get identical
    # executable outcomes; rows near the cut are completed from the label tail
    early = cut.ts_ns < cut.ts_ns.max() - 20 * S
    lab = ["ret_long_5", "ret_short_5", "entry_px_long", "lat_entry_ms"]
    np.testing.assert_array_equal(a.loc[early.to_numpy(), lab].to_numpy(), cut.loc[early, lab].to_numpy())
    assert cut[cut.ts_ns > cut.ts_ns.max() - 2 * S].ret_long_5.notna().all()


def test_label_frame_sanity(synth):
    d, s0 = synth
    df, meta = build_dataset(small_cfg(), d, s0 + 4 * 60 * S, s0 + 14 * 60 * S)
    v = df[df.valid & df.ret_long_5.notna() & df.ret_short_5.notna()]
    assert df.valid.mean() > 0.95
    # crossing the spread both ways and paying four fee legs must lose
    assert ((v.ret_long_5 + v.ret_short_5) < 0).mean() > 0.99
    assert (v.mae_long_5 <= 0).all() and (v.mae_short_5 <= 0).all()
    assert meta.latency["source"] == "prior"
    assert v.hold_ms_5.min() >= 5000
    # the planted lead-lag is visible to the cross-venue features
    assert np.corrcoef(v.xret_2, v.gross_long_5)[0, 1] > 0.1


def _book(venue, t, bid, ask, size=1.0):
    return BookSnapshot(venue, "X", t, t, [(bid, size), (bid - 1, 10.0)], [(ask, size), (ask + 1, 10.0)], 1)


def test_label_arithmetic_exact():
    """Hand-built sequence: one decision, deterministic latency 100ms."""
    cfg = Config()
    cfg.features.grid_ms = 500
    cfg.features.return_horizons_s = [0.5]
    cfg.features.flow_horizons_s = [0.5]
    cfg.labels.horizons_s = [1.0]
    cfg.labels.notional_usd = 100.0
    cfg.labels.passive = True
    cfg.labels.passive_wait_s = 1.0
    lat = LatencyModel(np.array([100.0]), "test")
    fees = lambda s: (0.001, 0.0)  # 10 bps taker, 0 maker
    t0 = 1_000 * S
    ev = [
        _book(LIGHTER, t0, 99.0, 101.0),
        BBO(REF, "X", t0, t0, 99.5, 1.0, 100.5, 1.0),
        MarketStats(LIGHTER, "X", t0, t0, funding_rate=0.0005, next_funding_ns=t0 + 200 * S),
    ]
    # keep both venues fresh every 100ms until the decision we label
    for k in range(1, 2500):
        t = t0 + k * 100_000_000
        ev.append(BBO(REF, "X", t, t, 99.5, 1.0, 100.5, 1.0))
        ev.append(BookDelta(LIGHTER, "X", t, t, [(99.0, 1.0)], [], 0, 0))
    # move the Lighter book at t0+240s: entry (decision ~t0+240.0 + 0.1s) sees 100/102
    t_move = t0 + 240 * S + 50_000_000
    ev.append(BookSnapshot(LIGHTER, "X", t_move, t_move, [(100.0, 2.0)], [(102.0, 2.0)], 2))
    t_exit_move = t0 + 241 * S
    ev.append(BookSnapshot(LIGHTER, "X", t_exit_move, t_exit_move, [(104.0, 2.0)], [(106.0, 2.0)], 3))
    ev.append(Trade(LIGHTER, "X", t_exit_move + 1, t_exit_move + 1, 104.0, 0.5, SELL))
    ev.sort(key=lambda e: e.ts_local_ns)
    start, end = t0 + 239 * S, t0 + 245 * S
    names, ts, syms, feats, info, labels = run_pipeline(iter(ev), ["X"], REF, cfg, lat, fees, start, end)
    df = assemble_frame(names, ts, syms, feats, info, labels)
    row = df[df.ts_ns == t0 + 240 * S + 500_000_000].iloc[0]  # decision at 240.5s
    # entry at 240.6s (book 100/102): long buys 100/102 = 0.98039 units at 102
    assert row.entry_px_long == pytest.approx(102.0)
    assert row.qty_long == pytest.approx(100 / 102)
    # exit at 240.6 + 1.0 + 0.1 = 241.7s (book 104/106): long sells at 104
    gross = (104.0 - 102.0) / 102.0
    assert row.gross_long_1 == pytest.approx(gross)
    assert row.ret_long_1 == pytest.approx(gross - 0.001 * (102 + 104) / 102)
    # short sold at 100, buys back at 106
    assert row.ret_short_1 == pytest.approx((100 - 106) / 100 - 0.001 * (100 + 106) / 100)
    # long MAE: marks on ticks between entry and exit; bid 100 then 104 -> 100/102-1
    assert row.mae_long_1 == pytest.approx(100 / 102 - 1)
    # passive long rests at 100 (bid at entry); sell trades at 104 never reach it
    assert row.pfill_long == 0.0
    # the short side rests at 102; book moves through it but no buy taker prints -> unfilled
    assert row.pfill_short == 0.0
    assert row.hold_ms_1 == pytest.approx(1100.0)


def test_funding_applied_when_crossed():
    cfg = Config()
    cfg.features.grid_ms = 500
    cfg.features.return_horizons_s = [0.5]
    cfg.features.flow_horizons_s = [0.5]
    cfg.labels.horizons_s = [1.0]
    cfg.labels.notional_usd = 1.0
    cfg.labels.passive = False
    lat = LatencyModel(np.array([100.0]), "test")
    t0 = 1_000 * S
    fund_t = t0 + 240 * S + 900_000_000
    ev = [_book(LIGHTER, t0, 99.0, 101.0), BBO(REF, "X", t0, t0, 99.5, 1.0, 100.5, 1.0),
          MarketStats(LIGHTER, "X", t0, t0, funding_rate=0.001, next_funding_ns=fund_t)]
    for k in range(1, 2500):
        t = t0 + k * 100_000_000
        ev.append(BBO(REF, "X", t, t, 99.5, 1.0, 100.5, 1.0))
        ev.append(BookDelta(LIGHTER, "X", t, t, [(99.0, 1.0)], [], 0, 0))
    names, ts, syms, feats, info, labels = run_pipeline(iter(ev), ["X"], REF, cfg, lat, lambda s: (0.0, 0.0),
                                                        t0 + 239 * S, t0 + 245 * S)
    df = assemble_frame(names, ts, syms, feats, info, labels)
    crossed = df[df.ts_ns == t0 + 240 * S].iloc[0]  # entry 240.1, exit 241.2 -> crosses 240.9
    after = df[df.ts_ns == t0 + 241 * S].iloc[0]
    assert crossed.fund_long_1 == pytest.approx(-0.001) and crossed.fund_short_1 == pytest.approx(0.001)
    assert after.fund_long_1 == 0.0


def test_day_chunks_are_seamless_and_cached(tmp_path):
    """A window crossing UTC midnight is built as two cached chunks with no
    missing ticks and labels completed across the boundary."""
    midnight = 1_767_312_000 * S  # 2026-01-02T00:00:00Z
    spec = SyntheticSpec(hours=0.5, seed=8, start_ns=midnight - 15 * 60 * S, symbols=[SymbolSpec()])
    generate(spec, str(tmp_path / "raw"))
    cfg = small_cfg()
    start, end = midnight - 10 * 60 * S, midnight + 10 * 60 * S
    import time as _t

    t0 = _t.time()
    df, meta = build_dataset(cfg, str(tmp_path / "raw"), start, end, cache_dir=str(tmp_path / "cache"))
    first = _t.time() - t0
    assert len(meta.extra["chunk_cache_keys"]) == 2
    ts = np.sort(df.ts_ns.unique())
    assert ts[0] == start and ts[-1] == end - cfg.features.grid_ms * 10**6
    assert np.all(np.diff(ts) == cfg.features.grid_ms * 10**6)  # no gap at midnight
    before = df[(df.ts_ns < midnight) & (df.ts_ns > midnight - 2 * S)]
    assert before.valid.all() and before.ret_long_5.notna().all()
    t0 = _t.time()
    df2, _ = build_dataset(cfg, str(tmp_path / "raw"), start, end, cache_dir=str(tmp_path / "cache"))
    assert _t.time() - t0 < first / 3
    np.testing.assert_array_equal(df2[meta.feature_names].to_numpy(), df[meta.feature_names].to_numpy())

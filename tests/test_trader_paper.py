"""Paper-mode trader driven by a replayed (synthetic) recording.

Exercises the live code path end to end without network: feature engine ->
decision engine -> allocator -> execution optimizer -> order manager ->
paper gateway fills -> persistence -> edge monitor, then a restart against
the same state database.
"""

import asyncio

import pytest

from apexmind.config import Config
from apexmind.core.clock import ManualClock
from apexmind.data.dataset import build_dataset
from apexmind.data.replay import Replay
from apexmind.data.synthetic import SymbolSpec, SyntheticSpec, generate
from apexmind.lab.laboratory import fit_production_bundle
from apexmind.lab.registry import Registry
from apexmind.live.state import StateStore
from apexmind.live.trader import PreflightError, Trader, live_preflight
from apexmind.venues.lighter.markets import LighterMarket

S = 10**9


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("paper")
    spec = SyntheticSpec(hours=2.0, seed=9, symbols=[SymbolSpec(), SymbolSpec(symbol="SOL", market_id=1, price=150.0,
                                                                               tick=0.01)])
    generate(spec, str(d / "raw"))
    cfg = Config()
    cfg.labels.latency_source = "prior"
    cfg.research.block_minutes = 2.0  # enough clusters in a short calibration window
    cfg.live.state_db = str(d / "state.sqlite")
    cfg.live.kill_file = str(d / "KILL")
    cfg.lab.registry_dir = str(d / "registry")
    s0 = spec.start_ns
    df, meta = build_dataset(cfg, str(d / "raw"), s0 + 5 * 60 * S, s0 + 60 * 60 * S)
    bundle = fit_production_bundle(cfg, df, meta, "ridge", "optimizer", {"hypothesis": "cvifa"})
    reg = Registry(cfg.lab.registry_dir)
    bid = reg.save(bundle)
    return cfg, d, s0, bid, reg, meta


async def drive(tr: Trader, clock: ManualClock, events) -> None:
    grid = tr.grid_ns
    for ev in events:
        if tr.next_tick is None:
            tr.next_tick = (ev.ts_local_ns // grid + 1) * grid
        while tr.next_tick < ev.ts_local_ns:
            clock.set(tr.next_tick)
            await tr.tick(tr.next_tick)
            tr.next_tick += grid
        clock.set(ev.ts_local_ns)
        tr.on_events([ev])


def make_trader(cfg, bid, reg, meta, clock):
    bundle = reg.load(bid)
    markets = {s: LighterMarket(**m) for s, m in meta.markets.items()}
    return Trader(cfg, bid, bundle, markets, set(markets), "binance_usdm", "paper", clock)


def test_paper_trading_end_to_end_and_restart(setup):
    cfg, d, s0, bid, reg, meta = setup
    clock = ManualClock(s0)
    tr = make_trader(cfg, bid, reg, meta, clock)
    events = Replay(str(d / "raw")).events(s0 + 60 * 60 * S - 10**6, s0 + 80 * 60 * S)
    asyncio.run(drive(tr, clock, events))
    store = tr.store
    closed = store.closed_positions()
    assert len(closed) > 20, tr.health()
    filled_entries = [p for p in closed if p["entry_notional"]]
    assert filled_entries and all(p["exit_price"] for p in filled_entries)
    n_dec = store.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n_dec >= len(filled_entries)
    assert store.last_equity() is not None and tr.equity != cfg.live.paper_equity
    # every order is accounted for: filled, cancelled or rejected (or still working at the cut)
    statuses = {r[0] for r in store.db.execute("SELECT DISTINCT status FROM orders")}
    assert statuses <= {"filled", "canceled", "rejected", "sent", "open", "partially_filled"}
    # positions never exceed the label notional the research validated
    mx = store.db.execute("SELECT MAX(entry_notional) FROM positions").fetchone()[0]
    assert mx <= cfg.labels.notional_usd * 1.01
    open_before = store.positions(("open", "closing"))
    edge_state = store.get(f"edge:{tr.bundle.name}")
    assert edge_state is not None
    tr.store.close()

    # --- restart: a new process picks up persisted positions and exits them ---
    clock2 = ManualClock(clock.now_ns())
    tr2 = make_trader(cfg, bid, reg, meta, clock2)
    assert tr2.edge.state()["n"] == edge_state["n"]
    # resume just before the next book snapshot, as a fresh subscription would
    events2 = Replay(str(d / "raw")).events(s0 + 90 * 60 * S - 10**6, s0 + 92 * 60 * S)
    asyncio.run(drive(tr2, clock2, events2))
    exits = tr2.store.db.execute("SELECT position_id, COUNT(*) FROM orders WHERE purpose='exit' "
                                 "GROUP BY position_id").fetchall()
    assert max(n for _, n in exits) <= 12  # backoff: no exit-order spam
    for p in open_before:
        row = tr2.store.db.execute("SELECT status FROM positions WHERE position_id=?", (p["position_id"],)).fetchone()
        assert row[0] == "closed"


def test_kill_file_flattens_and_blocks_entries(setup):
    cfg, d, s0, bid, reg, meta = setup
    clock = ManualClock(s0)
    tr = make_trader(cfg, bid, reg, meta, clock)
    (d / "KILL").write_text("stop")
    try:
        before = tr.store.db.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        events = Replay(str(d / "raw")).events(s0 + 60 * 60 * S - 10**6, s0 + 64 * 60 * S)
        asyncio.run(drive(tr, clock, events))
        after = tr.store.db.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        assert after == before
        assert tr.health()["halted"] == "kill_file"
    finally:
        (d / "KILL").unlink()


def test_live_preflight_refuses_unvalidated_bundle(setup):
    cfg, d, s0, bid, reg, meta = setup
    with pytest.raises(PreflightError) as e:
        live_preflight(cfg, reg.manifest(bid))
    msg = str(e.value)
    assert "synthetic" in msg and "measured execution latency" in msg and "integration test" in msg


def test_state_store_client_order_index_monotonic(tmp_path):
    s = StateStore(tmp_path / "s.sqlite")
    a = [s.next_client_order_index() for _ in range(5)]
    s.close()
    s2 = StateStore(tmp_path / "s.sqlite")
    b = s2.next_client_order_index()
    assert a == sorted(set(a)) and b > a[-1]

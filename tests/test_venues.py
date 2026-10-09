import asyncio
import json
import math

import pytest
import websockets

from apexmind.collector.quality import QualityMonitor
from apexmind.collector.recorder import RawRecorder, iter_raw
from apexmind.config import InstrumentConfig
from apexmind.core.clock import Clock, ManualClock
from apexmind.core.events import BBO, BUY, SELL, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade
from apexmind.core.orderbook import L2Book
from apexmind.data.replay import Replay
from apexmind.venues.binance.parse import BinanceParser
from apexmind.venues.common import to_ns
from apexmind.venues.feeds import LighterFeed
from apexmind.venues.lighter.markets import LighterMarket, parse_order_book_details
from apexmind.venues.lighter.parse import LighterParser
from apexmind.venues.mapping import build_instruments

DETAIL = {
    "symbol": "ETH", "market_id": 0, "market_type": "perp", "status": "active", "taker_fee": "0.0200",
    "maker_fee": "0.0020", "min_base_amount": "0.0050", "min_quote_amount": "10.000000", "size_decimals": 4,
    "price_decimals": 2, "default_initial_margin_fraction": 500, "min_initial_margin_fraction": 200,
    "maintenance_margin_fraction": 120, "closeout_margin_fraction": 80, "last_trade_price": 3000.5,
    "daily_quote_token_volume": 5e8, "open_interest": 1000.0,
}


def test_to_ns_units():
    assert to_ns(1_700_000_000) == 1_700_000_000 * 10**9
    assert to_ns(1_700_000_000_123) == 1_700_000_000_123 * 10**6
    assert to_ns(1_700_000_000_123_456) == 1_700_000_000_123_456 * 10**3
    assert to_ns(1_700_000_000_123_456_789) == 1_700_000_000_123_456_789
    assert to_ns(None) == 0


def test_market_detail_and_encoding():
    m = LighterMarket.from_detail(DETAIL)
    assert m.taker_fee == pytest.approx(0.0002) and m.maker_fee == pytest.approx(0.00002)
    assert m.default_imf == pytest.approx(0.05) and m.max_leverage == pytest.approx(50.0)
    assert m.mmf == pytest.approx(0.012)
    assert m.price_to_int(3000.567, side_is_buy=True) == 300056
    assert m.price_to_int(3000.561, side_is_buy=False) == 300057
    assert m.size_to_int(0.12349) == 1234
    assert m.quantize_size(0.12349) == pytest.approx(0.1234)
    # quote minimum dominates at this price: 10 / 3000 = 0.00333 < 0.005 base min
    assert m.min_feasible_qty(3000.0) == pytest.approx(0.005)
    assert m.min_feasible_qty(1000.0) == pytest.approx(0.01)
    assert m.order_is_feasible(0.005, 3000.0)
    assert not m.order_is_feasible(0.004, 3000.0)
    parsed = parse_order_book_details({"order_book_details": [DETAIL, {**DETAIL, "market_id": 9, "market_type": "spot"}]})
    assert list(parsed) == [0]


def lighter_msgs():
    snap = {"type": "subscribed/order_book", "channel": "order_book:0", "timestamp": 1_700_000_000_000,
            "order_book": {"asks": [{"price": "3001.00", "size": "1.5"}], "bids": [{"price": "2999.00", "size": "2.0"}],
                           "nonce": 10}}
    upd = {"type": "update/order_book", "channel": "order_book:0", "timestamp": 1_700_000_000_100,
           "order_book": {"asks": [{"price": "3001.00", "size": "0"}, {"price": "3000.50", "size": "0.7"}],
                          "bids": [], "nonce": 12, "begin_nonce": 10}}
    bad = {"type": "update/order_book", "channel": "order_book:0", "timestamp": 1_700_000_000_200,
           "order_book": {"asks": [], "bids": [], "nonce": 20, "begin_nonce": 15}}
    return snap, upd, bad


def test_lighter_book_continuity():
    p = LighterParser({0: "ETH"})
    snap, upd, bad = lighter_msgs()
    (s,) = p.parse(json.dumps(snap), 1)
    assert isinstance(s, BookSnapshot) and s.symbol == "ETH" and s.seq == 10
    (d,) = p.parse(json.dumps(upd), 2)
    assert isinstance(d, BookDelta) and d.asks == [(3001.0, 0.0), (3000.5, 0.7)]
    (g,) = p.parse(json.dumps(bad), 3)
    assert isinstance(g, FeedStatus) and g.status == "gap"
    # further deltas are refused until a fresh snapshot arrives
    (g2,) = p.parse(json.dumps(upd), 4)
    assert isinstance(g2, FeedStatus)
    book = L2Book()
    book.apply_snapshot(s.bids, s.asks)
    book.apply_delta(d.bids, d.asks)
    assert book.best_ask() == (3000.5, 0.7)


def test_lighter_trades_side_and_own_fill():
    p = LighterParser({0: "ETH"}, account_index=42)
    msg = {"type": "update/trade", "channel": "trade:0", "trades": [
        {"trade_id": 1, "market_id": 0, "price": "3000", "size": "0.1", "is_maker_ask": True,
         "timestamp": 1_700_000_000_000, "bid_account_id": 42, "ask_account_id": 7},
        {"trade_id": 2, "market_id": 0, "price": "2999", "size": "0.2", "is_maker_ask": False,
         "timestamp": 1_700_000_000_001, "bid_account_id": 8, "ask_account_id": 9},
    ]}
    t1, t2 = p.parse(json.dumps(msg), 5)
    assert t1.taker_side == BUY and t2.taker_side == SELL
    assert p.is_own_fill(t1) == BUY and p.is_own_fill(t2) == 0


def test_lighter_market_stats():
    p = LighterParser({0: "ETH"})
    msg = {"type": "update/market_stats", "channel": "market_stats:0", "market_stats": {
        "market_id": 0, "mark_price": "3000.1", "index_price": "3000.0", "current_funding_rate": "0.0012",
        "funding_timestamp": 1_700_003_600_000, "open_interest": "123"}}
    (s,) = p.parse(json.dumps(msg), 5)
    assert isinstance(s, MarketStats) and s.funding_rate == pytest.approx(0.000012)
    assert s.next_funding_ns == 1_700_003_600_000 * 10**6


def bn(stream, data):
    return json.dumps({"stream": stream, "data": data})


def depth(U, u, pu, b=(), a=()):
    return bn("btcusdt@depth@100ms", {"e": "depthUpdate", "E": 1, "T": 1, "s": "BTCUSDT", "U": U, "u": u, "pu": pu,
                                      "b": [list(x) for x in b], "a": [list(x) for x in a]})


def snapshot(last_id, bids, asks):
    return json.dumps({"stream": "rest:depth:BTCUSDT", "data": {"lastUpdateId": last_id, "E": 1, "T": 1,
                                                                 "bids": bids, "asks": asks}})


def test_binance_depth_sync_happy_path():
    p = BinanceParser({"BTCUSDT": "BTC"})
    assert p.parse(depth(90, 95, 89, b=[("100", "1")]), 1) == []  # buffered
    assert p.parse(depth(96, 105, 95, b=[("101", "2")]), 2) == []  # buffered, straddles snapshot
    out = p.parse(snapshot(100, [["100", "5"]], [["102", "1"]]), 3)
    assert isinstance(out[0], BookSnapshot)
    assert len(out) == 2 and isinstance(out[1], BookDelta) and out[1].seq_end == 105
    (d,) = p.parse(depth(106, 110, 105, a=[("102", "0")]), 4)
    assert isinstance(d, BookDelta)
    (g,) = p.parse(depth(115, 120, 112), 5)
    assert isinstance(g, FeedStatus) and g.status == "gap"
    assert p.needs_snapshot("BTCUSDT")


def test_binance_snapshot_not_bridged_by_buffer_is_gap():
    p = BinanceParser({"BTCUSDT": "BTC"})
    p.parse(depth(120, 130, 119), 1)
    out = p.parse(snapshot(100, [], []), 2)
    assert isinstance(out[-1], FeedStatus) and out[-1].status == "gap"


def test_binance_first_update_after_empty_buffer_must_bridge():
    p = BinanceParser({"BTCUSDT": "BTC"})
    p.parse(snapshot(100, [["1", "1"]], [["2", "1"]]), 1)
    assert p.parse(depth(90, 99, 89), 2) == []  # old, dropped
    (d,) = p.parse(depth(98, 103, 97), 3)  # straddles 100
    assert isinstance(d, BookDelta)


def test_binance_bbo_trade_mark_with_multiplier():
    p = BinanceParser({"1000PEPEUSDT": "PEPE"}, multipliers={"1000PEPEUSDT": 1e-3})
    (b,) = p.parse(bn("1000pepeusdt@bookTicker", {"e": "bookTicker", "u": 1, "E": 1, "T": 2, "s": "1000PEPEUSDT",
                                                   "b": "0.0100", "B": "10", "a": "0.0101", "A": "20"}), 1)
    assert isinstance(b, BBO) and b.bid_px == pytest.approx(1e-5) and b.bid_sz == pytest.approx(1e4)
    (t,) = p.parse(bn("x", {"e": "aggTrade", "E": 1, "a": 5, "s": "1000PEPEUSDT", "p": "0.0100", "q": "3",
                            "T": 2, "m": True}), 2)
    assert isinstance(t, Trade) and t.taker_side == SELL and t.price == pytest.approx(1e-5)
    (m,) = p.parse(bn("x", {"e": "markPriceUpdate", "E": 1, "s": "1000PEPEUSDT", "p": "0.0100", "i": "0.0100",
                            "r": "0.0001", "T": 1_700_000_000_000}), 3)
    assert m.funding_rate == pytest.approx(1e-4)
    assert p.parse(bn("x", {"e": "bookTicker", "s": "UNMAPPED", "b": "1", "B": "1", "a": "1", "A": "1"}), 4) == []


def test_instrument_mapping_multiplier_and_rejections():
    ms = parse_order_book_details({"order_book_details": [
        DETAIL,
        {**DETAIL, "symbol": "PEPE", "market_id": 1, "last_trade_price": 0.00001},
        {**DETAIL, "symbol": "FOO", "market_id": 2, "last_trade_price": 5.0},
        {**DETAIL, "symbol": "BTC", "market_id": 3, "last_trade_price": 60000.0},
    ]})
    cfg = InstrumentConfig(initial_bases=["ETH", "PEPE", "FOO"])
    prices = {"ETHUSDT": 3001.0, "1000PEPEUSDT": 0.01, "FOOUSDT": 9.0, "BTCUSDT": 60010.0}
    inst = {i.symbol: i for i in build_instruments(ms, prices, cfg, "binance_usdm")}
    assert inst["ETH"].eligible and inst["ETH"].price_multiplier == 1.0
    assert inst["PEPE"].ref_symbol == "1000PEPEUSDT" and inst["PEPE"].price_multiplier == 1e-3
    assert not inst["FOO"].eligible and "mismatch" in inst["FOO"].reason
    assert not inst["BTC"].eligible and "initial coverage" in inst["BTC"].reason


def test_recorder_roundtrip_and_replay(tmp_path):
    clk = ManualClock(1_700_000_000 * 10**9)
    rec = RawRecorder(tmp_path, clk, session_id="s1")
    rec.write_meta("lighter_markets", {"order_book_details": [DETAIL]})
    rec.write_meta("instruments", [{"symbol": "ETH", "market_id": 0, "ref_venue": "binance_usdm",
                                    "ref_symbol": "ETHUSDT", "price_multiplier": 1.0, "eligible": True,
                                    "reason": "ok"}])
    snap, upd, _ = lighter_msgs()
    t0 = clk.now_ns()
    rec.write("lighter", "c0", t0 + 10, json.dumps(snap))
    rec.write("binance_usdm", "c1", t0 + 5, bn("ethusdt@bookTicker", {"e": "bookTicker", "u": 1, "E": 1, "T": 1,
                                                                       "s": "ETHUSDT", "b": "2999", "B": "1",
                                                                       "a": "3001", "A": "1"}))
    rec.write("lighter", "c0", t0 + 20, json.dumps(upd))
    rec.close()
    rows = list(iter_raw(tmp_path, ["lighter", "binance_usdm"]))
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    assert rows[0][1] == "meta"
    rp = Replay(tmp_path)
    evs = list(rp.events())
    kinds = [type(e).__name__ for e in evs]
    assert kinds == ["BBO", "BookSnapshot", "BookDelta"]
    assert evs[0].symbol == "ETH" and evs[1].symbol == "ETH"
    q = QualityMonitor()
    for e in evs:
        q.observe(e)
    rep = q.report()
    assert rep["lighter:ETH"]["snapshots"] == 1 and rep["lighter:ETH"]["book_updates"] == 1


async def test_lighter_feed_against_local_server():
    """Exercise subscribe, ping/pong and parsing against a local WS server."""
    snap, upd, _ = lighter_msgs()
    got_pong = asyncio.Event()
    subs = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "connected"}))
        while len(subs) < 3:
            subs.append(json.loads(await ws.recv())["channel"])
        await ws.send(json.dumps({"type": "ping"}))
        msg = json.loads(await ws.recv())
        if msg.get("type") == "pong":
            got_pong.set()
        await ws.send(json.dumps(snap))
        await ws.send(json.dumps(upd))
        await asyncio.sleep(0.5)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        raws, events = [], []
        feed = LighterFeed(f"ws://127.0.0.1:{port}", [0], LighterParser({0: "ETH"}), Clock(),
                           lambda *a: raws.append(a), events.extend)
        stop = asyncio.Event()
        task = asyncio.create_task(feed.run(stop))
        for _ in range(100):
            if any(isinstance(e, BookDelta) for e in events):
                break
            await asyncio.sleep(0.05)
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert subs == ["order_book/0", "trade/0", "market_stats/0"]
    assert got_pong.is_set()
    assert any(isinstance(e, BookSnapshot) for e in events) and any(isinstance(e, BookDelta) for e in events)
    assert all(r[2] > 0 for r in raws)

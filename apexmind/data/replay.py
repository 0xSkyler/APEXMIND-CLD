"""Turn recorded raw messages back into normalized events.

Uses exactly the parser classes the live feeds use. Market metadata and
instrument mappings are taken from the ``meta`` records written by the
collector at the time, never from today's exchange state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from apexmind.collector.recorder import META, iter_raw
from apexmind.core.events import FeedStatus
from apexmind.venues.binance.parse import VENUE as BINANCE, BinanceParser
from apexmind.venues.lighter.markets import LighterMarket, parse_order_book_details
from apexmind.venues.lighter.parse import LighterParser
from apexmind.venues.mapping import Instrument


class Replay:
    def __init__(self, data_dir: str | os.PathLike, ref_venue: str = BINANCE, fee_unit: str = "percent",
                 account_index: int = -1) -> None:
        self.data_dir = data_dir
        self.ref_venue = ref_venue
        self.fee_unit = fee_unit
        self.lighter = LighterParser({}, account_index)
        self.ref = BinanceParser({})
        self.markets: dict[int, LighterMarket] = {}
        self.instruments: dict[str, Instrument] = {}
        self.exec_records: list[dict] = []
        self.clock_probes: list[dict] = []
        self.sessions: list[dict] = []

    def _meta(self, raw: str, t: int) -> list:
        msg = json.loads(raw)
        kind, data = msg.get("type"), msg.get("data")
        if kind == "lighter_markets":
            self.markets = parse_order_book_details(data, self.fee_unit)
            self.lighter.symbols = {m.market_id: m.symbol for m in self.markets.values()}
        elif kind == "instruments":
            self.instruments = {d["symbol"]: Instrument(**d) for d in data}
            elig = [i for i in self.instruments.values() if i.ref_symbol]
            self.ref.set_mapping({i.ref_symbol: i.symbol for i in elig},
                                 {i.ref_symbol: i.price_multiplier for i in elig})
        elif kind == "exec":
            self.exec_records.append({"t": t, **data})
        elif kind == "clock_probe":
            self.clock_probes.append({"t": t, **data})
        elif kind == "session":
            self.sessions.append({"t": t, **data})
        return []

    def events(self, start_ns: int = 0, end_ns: int = 2**63 - 1) -> Iterator:
        # Metadata written before the window still applies inside it.
        if start_ns > 0:
            for t, _v, _c, raw in iter_raw(self.data_dir, [], 0, start_ns):
                self._meta(raw, t)
        for t, venue, _conn, raw in iter_raw(self.data_dir, ["lighter", self.ref_venue], start_ns, end_ns):
            if venue == META:
                self._meta(raw, t)
                continue
            if '"_feed/' in raw[:40]:
                status = "connected" if "connected" in raw[:40] else "disconnected"
                if venue == "lighter":
                    self.lighter.reset_all()
                else:
                    for s in list(self.ref.symbols):
                        self.ref.reset(s)
                yield FeedStatus(venue, "*", t, status, "replayed")
                continue
            parser = self.lighter if venue == "lighter" else self.ref
            yield from parser.parse(raw, t)

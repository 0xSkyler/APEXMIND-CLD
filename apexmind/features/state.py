"""Point-in-time market state shared by features, labels and live trading.

Events must be applied in local-arrival order. Everything here is a
cumulative or latest-value quantity, so readers can take differences over
any window without the state needing to know the window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from apexmind.core.clock import RollingQuantiles
from apexmind.core.events import BBO, BUY, LIGHTER, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade
from apexmind.core.orderbook import L2Book


@dataclass
class VenueState:
    book: L2Book = field(default_factory=L2Book)
    has_bbo_stream: bool = False
    bbo: tuple[float, float, float, float] | None = None  # bid_px, bid_sz, ask_px, ask_sz
    last_book_ns: int = 0  # local time of last top-of-book information
    last_trade_ns: int = 0
    last_any_ns: int = 0
    # cumulative trade flow (base units) and counts
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    trades: int = 0
    # cumulative L1 order-flow imbalance (Cont, Kukanov & Stoikov 2014)
    ofi: float = 0.0
    _l1: tuple[float, float, float, float] | None = None
    # cumulative size added / removed at levels within the depth band
    added: float = 0.0
    removed: float = 0.0
    funding_rate: float = math.nan
    next_funding_ns: int = 0
    mark_price: float = math.nan
    delay_ms: RollingQuantiles = field(default_factory=lambda: RollingQuantiles(512))
    gaps: int = 0

    def top(self) -> tuple[float, float, float, float] | None:
        """Best bid/ask: dedicated BBO stream when present, else the book."""
        if self.has_bbo_stream and self.bbo is not None:
            return self.bbo
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if bb is None or ba is None:
            return None
        return bb[0], bb[1], ba[0], ba[1]

    def mid(self) -> float:
        t = self.top()
        return math.nan if t is None else 0.5 * (t[0] + t[2])

    def usable(self) -> bool:
        if self.has_bbo_stream:
            return self.bbo is not None and self.bbo[0] < self.bbo[2]
        return self.book.usable()

    def update_ofi(self) -> None:
        t = self.top()
        if t is None:
            return
        if self._l1 is not None:
            pb0, qb0, pa0, qa0 = self._l1
            pb, qb, pa, qa = t
            e = 0.0
            if pb >= pb0:
                e += qb
            if pb <= pb0:
                e -= qb0
            if pa <= pa0:
                e -= qa
            if pa >= pa0:
                e += qa0
            self.ofi += e
        self._l1 = t


class MarketState:
    def __init__(self, symbols: list[str], ref_venue: str, depth_band_bps: float = 10.0) -> None:
        self.ref_venue = ref_venue
        self.band = depth_band_bps * 1e-4
        self.lit: dict[str, VenueState] = {s: VenueState() for s in symbols}
        self.ref: dict[str, VenueState] = {s: VenueState() for s in symbols}
        self.now_ns = 0

    def venue_state(self, venue: str, symbol: str) -> VenueState | None:
        table = self.lit if venue == LIGHTER else self.ref if venue == self.ref_venue else None
        return None if table is None else table.get(symbol)

    def _band_flow(self, vs: VenueState, side_levels, levels, is_bid: bool) -> None:
        mid = vs.book.mid()
        if math.isnan(mid):
            return
        lo, hi = mid * (1 - self.band), mid * (1 + self.band)
        for p, s in levels:
            if not (lo <= p <= hi):
                continue
            old = side_levels.get(p, 0.0)
            if s > old:
                vs.added += s - old
            else:
                vs.removed += old - s

    def apply(self, ev) -> None:
        self.now_ns = max(self.now_ns, ev.ts_local_ns)
        if isinstance(ev, FeedStatus):
            targets = []
            if ev.symbol == "*":
                table = self.lit if ev.venue == LIGHTER else self.ref if ev.venue == self.ref_venue else {}
                targets = list(table.values())
            else:
                vs = self.venue_state(ev.venue, ev.symbol)
                targets = [vs] if vs else []
            if ev.status in ("gap", "disconnected"):
                for vs in targets:
                    vs.book.invalidate()
                    vs.gaps += 1
                    if ev.status == "disconnected":
                        vs.bbo = None
            return
        vs = self.venue_state(ev.venue, ev.symbol)
        if vs is None:
            return
        vs.last_any_ns = ev.ts_local_ns
        if ev.ts_exch_ns and not isinstance(ev, (Trade, MarketStats)):
            vs.delay_ms.add((ev.ts_local_ns - ev.ts_exch_ns) / 1e6)
        if isinstance(ev, BookDelta):
            if vs.book.valid:
                self._band_flow(vs, vs.book.bids, ev.bids, True)
                self._band_flow(vs, vs.book.asks, ev.asks, False)
            vs.book.apply_delta(ev.bids, ev.asks, ev.seq_end)
            if not vs.has_bbo_stream:
                vs.last_book_ns = ev.ts_local_ns
                vs.update_ofi()
        elif isinstance(ev, BBO):
            vs.has_bbo_stream = True
            vs.bbo = (ev.bid_px, ev.bid_sz, ev.ask_px, ev.ask_sz)
            vs.last_book_ns = ev.ts_local_ns
            vs.update_ofi()
        elif isinstance(ev, Trade):
            if ev.taker_side == BUY:
                vs.buy_vol += ev.size
            else:
                vs.sell_vol += ev.size
            vs.trades += 1
            vs.last_trade_ns = ev.ts_local_ns
        elif isinstance(ev, BookSnapshot):
            vs.book.apply_snapshot(ev.bids, ev.asks, ev.seq)
            if not vs.has_bbo_stream:
                vs.last_book_ns = ev.ts_local_ns
                vs._l1 = None  # no OFI across a discontinuity
                vs.update_ofi()
        elif isinstance(ev, MarketStats):
            if not math.isnan(ev.funding_rate):
                vs.funding_rate = ev.funding_rate
            if ev.next_funding_ns:
                vs.next_funding_ns = ev.next_funding_ns
            if not math.isnan(ev.mark_price):
                vs.mark_price = ev.mark_price

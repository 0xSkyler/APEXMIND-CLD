"""Level-2 order book with executable-price walks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

from sortedcontainers import SortedDict

from apexmind.core.events import BUY


@dataclass(slots=True)
class Fill:
    vwap: float
    qty: float
    notional: float
    worst_price: float
    levels: int
    complete: bool  # False when the visible book could not absorb the order


class L2Book:
    __slots__ = ("bids", "asks", "seq", "ts_local_ns", "ts_exch_ns", "valid", "updates")

    def __init__(self) -> None:
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()
        self.seq = 0
        self.ts_local_ns = 0
        self.ts_exch_ns = 0
        self.valid = False
        self.updates = 0

    # -- mutation -----------------------------------------------------------
    def apply_snapshot(self, bids, asks, seq: int = 0) -> None:
        self.bids = SortedDict({p: s for p, s in bids if s > 0})
        self.asks = SortedDict({p: s for p, s in asks if s > 0})
        self.seq = seq
        self.valid = True
        self.updates += 1

    def apply_delta(self, bids, asks, seq: int = 0) -> None:
        for p, s in bids:
            if s > 0:
                self.bids[p] = s
            else:
                self.bids.pop(p, None)
        for p, s in asks:
            if s > 0:
                self.asks[p] = s
            else:
                self.asks.pop(p, None)
        if seq:
            self.seq = seq
        self.updates += 1

    def set_bbo(self, bid_px: float, bid_sz: float, ask_px: float, ask_sz: float) -> None:
        """Overwrite the top of book from a BBO stream.

        Levels better than the new touch are stale and removed; deeper levels
        are kept so depth features remain available between diff updates.
        """
        for p in [p for p in self.bids.irange(minimum=bid_px, inclusive=(False, True))]:
            self.bids.pop(p)
        for p in [p for p in self.asks.irange(maximum=ask_px, inclusive=(True, False))]:
            self.asks.pop(p)
        if bid_sz > 0:
            self.bids[bid_px] = bid_sz
        if ask_sz > 0:
            self.asks[ask_px] = ask_sz
        self.updates += 1

    def invalidate(self) -> None:
        self.valid = False

    # -- queries --------------------------------------------------------------
    def best_bid(self) -> tuple[float, float] | None:
        return self.bids.peekitem(-1) if self.bids else None

    def best_ask(self) -> tuple[float, float] | None:
        return self.asks.peekitem(0) if self.asks else None

    def mid(self) -> float:
        if not self.bids or not self.asks:
            return math.nan
        return 0.5 * (self.bids.peekitem(-1)[0] + self.asks.peekitem(0)[0])

    def microprice(self) -> float:
        if not self.bids or not self.asks:
            return math.nan
        bp, bs = self.bids.peekitem(-1)
        ap, as_ = self.asks.peekitem(0)
        tot = bs + as_
        return 0.5 * (bp + ap) if tot <= 0 else (bp * as_ + ap * bs) / tot

    def spread(self) -> float:
        if not self.bids or not self.asks:
            return math.nan
        return self.asks.peekitem(0)[0] - self.bids.peekitem(-1)[0]

    def is_crossed(self) -> bool:
        return bool(self.bids and self.asks and self.bids.peekitem(-1)[0] >= self.asks.peekitem(0)[0])

    def usable(self) -> bool:
        return self.valid and bool(self.bids) and bool(self.asks) and not self.is_crossed()

    def _levels(self, side: int):
        """Iterate (price, size) from the touch outward; side=BUY means bids."""
        if side == BUY:
            d = self.bids
            return ((p, d[p]) for p in reversed(d))
        d = self.asks
        return ((p, d[p]) for p in d)

    def top(self, side: int, n: int) -> list[tuple[float, float]]:
        """Best ``n`` levels; side=BUY means bids."""
        return list(islice(self._levels(side), n))

    def depth_qty(self, side: int, n: int) -> float:
        return sum(s for _, s in self.top(side, n))

    def depth_notional_within(self, side: int, band_bps: float) -> float:
        mid = self.mid()
        if math.isnan(mid):
            return 0.0
        if side == BUY:
            lo = mid * (1 - band_bps * 1e-4)
            return sum(p * self.bids[p] for p in self.bids.irange(minimum=lo))
        hi = mid * (1 + band_bps * 1e-4)
        return sum(p * self.asks[p] for p in self.asks.irange(maximum=hi))

    def walk(self, taker_side: int, qty: float | None = None, notional: float | None = None) -> Fill:
        """Executable fill for an aggressive order.

        ``taker_side`` BUY consumes asks from the best upward; SELL consumes
        bids downward. Exactly one of ``qty``/``notional`` must be given.
        """
        if (qty is None) == (notional is None):
            raise ValueError("specify exactly one of qty or notional")
        levels = self._levels(-taker_side)
        rem_q = qty
        rem_n = notional
        got_q = 0.0
        got_n = 0.0
        worst = math.nan
        used = 0
        for p, s in levels:
            if rem_q is not None:
                take = min(s, rem_q)
                rem_q -= take
            else:
                take = min(s, rem_n / p)
                rem_n -= take * p
            if take <= 0:
                break
            got_q += take
            got_n += take * p
            worst = p
            used += 1
            if (rem_q is not None and rem_q <= 1e-15) or (rem_n is not None and rem_n <= 1e-9):
                break
        complete = (rem_q is not None and rem_q <= 1e-12) or (rem_n is not None and rem_n <= 1e-6)
        vwap = got_n / got_q if got_q > 0 else math.nan
        return Fill(vwap, got_q, got_n, worst, used, complete)

    def impact_bps(self, taker_side: int, notional: float) -> float:
        """Signed cost vs mid of executing ``notional`` now, in bps (>= 0)."""
        mid = self.mid()
        f = self.walk(taker_side, notional=notional)
        if math.isnan(mid) or not f.complete:
            return math.nan
        return (f.vwap / mid - 1.0) * 1e4 * taker_side

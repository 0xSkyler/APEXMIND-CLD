"""Lighter WebSocket/REST message -> normalized events.

Stateful only where the protocol requires it: market id -> symbol mapping
and order-book continuity (``begin_nonce`` of an update must equal the
``nonce`` of the previous one; otherwise the book is invalid until the next
snapshot, which the feed obtains by resubscribing).
"""

from __future__ import annotations

from apexmind.core.events import BUY, LIGHTER, SELL, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade
from apexmind.venues.common import levels, loads, to_ns


def channel_market(channel: str) -> int | None:
    """``order_book:3`` / ``order_book/3`` -> 3."""
    for sep in (":", "/"):
        if sep in channel:
            tail = channel.rsplit(sep, 1)[1]
            if tail.isdigit():
                return int(tail)
    return None


class LighterParser:
    def __init__(self, symbols: dict[int, str], account_index: int = -1) -> None:
        self.symbols = dict(symbols)
        self.account_index = account_index
        self._last_nonce: dict[int, int] = {}
        self.gaps = 0

    def reset_all(self) -> None:
        """Forget book continuity (connection lost; await fresh snapshots)."""
        self._last_nonce.clear()

    def _sym(self, market_id: int | None) -> str | None:
        if market_id is None:
            return None
        return self.symbols.get(market_id)

    def parse(self, raw, ts_local_ns: int) -> list:
        msg = loads(raw)
        mtype = msg.get("type", "")
        channel = msg.get("channel", "")
        if mtype.endswith("order_book"):
            return self._order_book(msg, mtype, channel, ts_local_ns)
        if mtype.endswith("/trade"):
            return self._trades(msg, channel, ts_local_ns)
        if mtype.endswith("market_stats"):
            return self._stats(msg, channel, ts_local_ns)
        return []

    def _order_book(self, msg, mtype, channel, ts_local_ns):
        mid = channel_market(channel)
        sym = self._sym(mid)
        if sym is None:
            return []
        ob = msg.get("order_book") or {}
        ts_exch = to_ns(msg.get("timestamp") or ob.get("timestamp"))
        bids, asks = levels(ob.get("bids")), levels(ob.get("asks"))
        nonce = int(ob.get("nonce", 0) or 0)
        begin = ob.get("begin_nonce")
        if mtype.startswith("subscribed"):
            self._last_nonce[mid] = nonce
            return [BookSnapshot(LIGHTER, sym, ts_exch, ts_local_ns, bids, asks, nonce)]
        out: list = []
        last = self._last_nonce.get(mid)
        if last is None:
            # Delta without a snapshot: unusable until resubscribed.
            return [FeedStatus(LIGHTER, sym, ts_local_ns, "gap", "delta before snapshot")]
        if begin is not None and nonce and int(begin) != last:
            self.gaps += 1
            self._last_nonce.pop(mid, None)
            return [FeedStatus(LIGHTER, sym, ts_local_ns, "gap", f"begin_nonce {begin} != last {last}")]
        if nonce:
            self._last_nonce[mid] = nonce
        out.append(BookDelta(LIGHTER, sym, ts_exch, ts_local_ns, bids, asks, int(begin or 0), nonce, last))
        return out

    def _trades(self, msg, channel, ts_local_ns):
        mid = channel_market(channel)
        out = []
        for t in msg.get("trades") or []:
            sym = self._sym(int(t.get("market_id", mid if mid is not None else -1)))
            if sym is None:
                continue
            # is_maker_ask: the resting order was the ask => the taker bought.
            taker = BUY if t.get("is_maker_ask") else SELL
            out.append(
                Trade(
                    LIGHTER,
                    sym,
                    to_ns(t.get("timestamp") or t.get("transaction_time")),
                    ts_local_ns,
                    float(t["price"]),
                    float(t["size"]),
                    taker,
                    str(t.get("trade_id", "")),
                    int(t.get("bid_account_id", -1)),
                    int(t.get("ask_account_id", -1)),
                )
            )
        return out

    def _stats(self, msg, channel, ts_local_ns):
        st = msg.get("market_stats") or {}
        rows = st.values() if st and all(isinstance(v, dict) for v in st.values()) else [st]
        out = []
        for s in rows:
            mid = s.get("market_id", channel_market(channel))
            sym = self._sym(int(mid)) if mid is not None else None
            if sym is None:
                continue
            fr = s.get("current_funding_rate", s.get("funding_rate"))
            out.append(
                MarketStats(
                    LIGHTER,
                    sym,
                    to_ns(msg.get("timestamp")),
                    ts_local_ns,
                    mark_price=float(s.get("mark_price") or "nan"),
                    index_price=float(s.get("index_price") or "nan"),
                    # Lighter reports funding rates in percent per interval.
                    funding_rate=float(fr) / 100.0 if fr not in (None, "") else float("nan"),
                    next_funding_ns=to_ns(s.get("funding_timestamp")),
                    open_interest=float(s.get("open_interest") or "nan"),
                )
            )
        return out

    def is_own_fill(self, t: Trade) -> int:
        """+1 if our account bought, -1 if it sold, 0 if not ours."""
        if self.account_index < 0:
            return 0
        if t.bid_account == self.account_index:
            return BUY
        if t.ask_account == self.account_index:
            return SELL
        return 0

"""Binance USD-M futures combined-stream messages -> normalized events.

Diff-depth synchronisation follows the venue's documented procedure:
buffer updates, take a REST snapshot (``lastUpdateId``), drop updates with
``u < lastUpdateId``, the first applied update must straddle the snapshot
(``U <= lastUpdateId <= u``) and every later update must satisfy
``pu == previous u``. Any violation invalidates the book until resync.

Snapshots are recorded as synthetic messages
``{"stream": "rest:depth:<SYMBOL>", "data": <REST payload>}`` so historical
replay reconstructs exactly the same book as the live process saw.
"""

from __future__ import annotations

from apexmind.core.events import BBO, BUY, SELL, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade
from apexmind.venues.common import levels, loads, to_ns

VENUE = "binance_usdm"


class _DepthSync:
    __slots__ = ("synced", "last_u", "buffer", "bridged")

    def __init__(self) -> None:
        self.synced = False
        self.bridged = False  # first update after the snapshot has been applied
        self.last_u = -1
        self.buffer: list[tuple[dict, int]] = []


class BinanceParser:
    def __init__(self, symbols: dict[str, str] | None = None, buffer_limit: int = 2000,
                 multipliers: dict[str, float] | None = None) -> None:
        # venue symbol (e.g. BTCUSDT) -> canonical symbol used internally
        self.symbols = {k.upper(): v for k, v in (symbols or {}).items()}
        # price multiplier into canonical (Lighter) units; sizes scale inversely
        self.multipliers = {k.upper(): float(v) for k, v in (multipliers or {}).items()}
        self._sync: dict[str, _DepthSync] = {}
        self.buffer_limit = buffer_limit
        self.gaps = 0

    def set_mapping(self, symbols: dict[str, str], multipliers: dict[str, float]) -> None:
        self.symbols = {k.upper(): v for k, v in symbols.items()}
        self.multipliers = {k.upper(): float(v) for k, v in multipliers.items()}

    def _canon(self, venue_sym: str) -> str | None:
        if not self.symbols:
            return venue_sym
        return self.symbols.get(venue_sym.upper())

    def _k(self, venue_sym: str) -> float:
        return self.multipliers.get(venue_sym.upper(), 1.0)

    def _lv(self, venue_sym: str, raw) -> list[tuple[float, float]]:
        k = self._k(venue_sym)
        lv = levels(raw)
        return lv if k == 1.0 else [(p * k, s / k) for p, s in lv]

    def needs_snapshot(self, venue_sym: str) -> bool:
        st = self._sync.get(venue_sym.upper())
        return st is None or not st.synced

    def reset(self, venue_sym: str) -> None:
        """Forget depth continuity (e.g. after a reconnect)."""
        self._sync.pop(venue_sym.upper(), None)

    def parse(self, raw, ts_local_ns: int) -> list:
        msg = loads(raw)
        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        if stream.startswith("rest:depth:"):
            return self._snapshot(stream.split(":", 2)[2], data, ts_local_ns)
        et = data.get("e")
        if et == "bookTicker":
            sym = self._canon(data["s"])
            if sym is None:
                return []
            k = self._k(data["s"])
            return [
                BBO(VENUE, sym, to_ns(data.get("T") or data.get("E")), ts_local_ns,
                    float(data["b"]) * k, float(data["B"]) / k, float(data["a"]) * k, float(data["A"]) / k,
                    int(data.get("u", 0)))
            ]
        if et == "depthUpdate":
            return self._depth(data, ts_local_ns)
        if et == "aggTrade":
            sym = self._canon(data["s"])
            if sym is None:
                return []
            # m == True: buyer is the maker => taker sold
            side = SELL if data.get("m") else BUY
            k = self._k(data["s"])
            return [Trade(VENUE, sym, to_ns(data.get("T") or data.get("E")), ts_local_ns,
                          float(data["p"]) * k, float(data["q"]) / k, side, str(data.get("a", "")))]
        if et == "markPriceUpdate":
            sym = self._canon(data["s"])
            if sym is None:
                return []
            k = self._k(data["s"])
            return [MarketStats(VENUE, sym, to_ns(data.get("E")), ts_local_ns,
                                mark_price=float(data.get("p") or "nan") * k,
                                index_price=float(data.get("i") or "nan") * k,
                                funding_rate=float(data.get("r") or "nan"),
                                next_funding_ns=to_ns(data.get("T")))]
        return []

    def _snapshot(self, venue_sym: str, data: dict, ts_local_ns: int) -> list:
        venue_sym = venue_sym.upper()
        sym = self._canon(venue_sym)
        if sym is None:
            return []
        st = self._sync.setdefault(venue_sym, _DepthSync())
        last_id = int(data["lastUpdateId"])
        out: list = [BookSnapshot(VENUE, sym, to_ns(data.get("T") or data.get("E")), ts_local_ns,
                                  self._lv(venue_sym, data.get("bids")), self._lv(venue_sym, data.get("asks")), last_id)]
        st.synced = True
        st.bridged = False
        st.last_u = last_id
        pending, st.buffer = st.buffer, []
        for i, (upd, _) in enumerate(pending):
            # Buffered deltas become usable only now, so they carry the
            # snapshot's arrival time.
            out.extend(self._apply(st, sym, upd, ts_local_ns))
            if not st.synced:
                st.buffer.extend(pending[i + 1 :])
                break
        return out

    def _apply(self, st: _DepthSync, sym: str, d: dict, ts_local_ns: int) -> list:
        U, u = int(d["U"]), int(d["u"])
        if u < st.last_u or (st.bridged and u <= st.last_u):
            return []  # already contained in the snapshot
        if not st.bridged:
            # Must contain or directly continue the snapshot id: an update
            # starting at lastUpdateId + 1 leaves nothing missing.
            if not (U <= st.last_u + 1 and u >= st.last_u):
                return self._gap(st, sym, d, ts_local_ns, f"first update [{U},{u}] does not bridge {st.last_u}")
            st.bridged = True
        elif int(d.get("pu", -1)) != st.last_u:
            return self._gap(st, sym, d, ts_local_ns, f"pu {d.get('pu')} != last u {st.last_u}")
        st.last_u = u
        return [self._delta_event(sym, d, ts_local_ns)]

    def _gap(self, st: _DepthSync, sym: str, d: dict, ts_local_ns: int, why: str) -> list:
        st.synced = False
        st.bridged = False
        st.buffer = [(d, ts_local_ns)]
        self.gaps += 1
        return [FeedStatus(VENUE, sym, ts_local_ns, "gap", why)]

    def _delta_event(self, sym: str, d: dict, ts_local_ns: int) -> BookDelta:
        return BookDelta(VENUE, sym, to_ns(d.get("T") or d.get("E")), ts_local_ns,
                         self._lv(d["s"], d.get("b")), self._lv(d["s"], d.get("a")), int(d["U"]), int(d["u"]),
                         int(d.get("pu", -1)))

    def _depth(self, d: dict, ts_local_ns: int) -> list:
        venue_sym = d["s"].upper()
        sym = self._canon(venue_sym)
        if sym is None:
            return []
        st = self._sync.setdefault(venue_sym, _DepthSync())
        if not st.synced:
            st.buffer.append((d, ts_local_ns))
            if len(st.buffer) > self.buffer_limit:
                st.buffer = st.buffer[-self.buffer_limit:]
            return []
        return self._apply(st, sym, d, ts_local_ns)

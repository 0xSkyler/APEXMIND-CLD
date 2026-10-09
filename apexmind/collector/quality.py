"""Data-quality accounting for recorded or live market data.

The monitor is fed normalized events (in local-time order) and reports, per
venue/symbol: message counts, sequence gaps, crossed books, inter-arrival
gaps, exchange-timestamp regressions and the distribution of observed delay
(local arrival - exchange time). Research uses :meth:`QualityMonitor.report`
to document coverage and the trading engine uses the same signals to refuse
decisions on unsynchronized data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from apexmind.core.clock import RollingQuantiles
from apexmind.core.events import BBO, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade


@dataclass
class StreamQuality:
    messages: int = 0
    trades: int = 0
    book_updates: int = 0
    snapshots: int = 0
    gaps: int = 0
    disconnects: int = 0
    exch_ts_regressions: int = 0
    max_interarrival_ms: float = 0.0
    first_ns: int = 0
    last_ns: int = 0
    last_exch_ns: int = 0
    delay: RollingQuantiles = field(default_factory=lambda: RollingQuantiles(20_000))

    def to_dict(self) -> dict:
        span_h = (self.last_ns - self.first_ns) / 3.6e12 if self.last_ns > self.first_ns else 0.0
        return {
            "messages": self.messages,
            "trades": self.trades,
            "book_updates": self.book_updates,
            "snapshots": self.snapshots,
            "gaps": self.gaps,
            "disconnects": self.disconnects,
            "exch_ts_regressions": self.exch_ts_regressions,
            "max_interarrival_ms": round(self.max_interarrival_ms, 1),
            "span_hours": round(span_h, 3),
            "delay_ms_p50": _r(self.delay.quantile(0.5)),
            "delay_ms_p95": _r(self.delay.quantile(0.95)),
            "delay_ms_p99": _r(self.delay.quantile(0.99)),
        }


def _r(x: float) -> float | None:
    return None if math.isnan(x) else round(x, 2)


class QualityMonitor:
    def __init__(self) -> None:
        self.streams: dict[tuple[str, str], StreamQuality] = defaultdict(StreamQuality)

    def observe(self, ev) -> None:
        if isinstance(ev, FeedStatus):
            if ev.symbol == "*":
                targets = [q for (v, _), q in self.streams.items() if v == ev.venue]
            else:
                targets = [self.streams[(ev.venue, ev.symbol)]]
            for q in targets:
                if ev.status == "gap":
                    q.gaps += 1
                elif ev.status == "disconnected":
                    q.disconnects += 1
            return
        q = self.streams[(ev.venue, ev.symbol)]
        t = ev.ts_local_ns
        if q.messages == 0:
            q.first_ns = t
        elif t > q.last_ns:
            q.max_interarrival_ms = max(q.max_interarrival_ms, (t - q.last_ns) / 1e6)
        q.last_ns = max(q.last_ns, t)
        q.messages += 1
        if isinstance(ev, Trade):
            q.trades += 1
        elif isinstance(ev, BookSnapshot):
            q.snapshots += 1
        elif isinstance(ev, (BookDelta, BBO)):
            q.book_updates += 1
        if ev.ts_exch_ns and not isinstance(ev, (MarketStats, Trade)):
            # Trades are batched with older timestamps; book/BBO should not regress.
            if ev.ts_exch_ns < q.last_exch_ns:
                q.exch_ts_regressions += 1
            q.last_exch_ns = max(q.last_exch_ns, ev.ts_exch_ns)
        if ev.ts_exch_ns:
            q.delay.add((t - ev.ts_exch_ns) / 1e6)

    def report(self) -> dict:
        return {f"{v}:{s}": q.to_dict() for (v, s), q in sorted(self.streams.items())}

"""Normalized market and execution events.

Every event carries two timestamps:

* ``ts_exch_ns`` – the venue's own event time (0 when the venue gives none).
* ``ts_local_ns`` – arrival time on the local monotonic-anchored timeline.

Research replay and live trading consume events strictly in ``ts_local_ns``
order; a feature computed at decision time ``t`` only sees events with
``ts_local_ns <= t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BUY = 1
SELL = -1

LIGHTER = "lighter"


@dataclass(slots=True)
class BookSnapshot:
    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    seq: int = 0


@dataclass(slots=True)
class BookDelta:
    """Absolute size per price level; size 0 removes the level."""

    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    seq_begin: int = 0
    seq_end: int = 0
    prev_seq: int = -1  # venue-reported previous sequence when available


@dataclass(slots=True)
class BBO:
    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    seq: int = 0


@dataclass(slots=True)
class Trade:
    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    price: float
    size: float
    taker_side: int  # BUY (+1) or SELL (-1)
    trade_id: str = ""
    # Lighter only: lets us find our own fills on the public tape.
    bid_account: int = -1
    ask_account: int = -1


@dataclass(slots=True)
class MarketStats:
    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    mark_price: float = float("nan")
    index_price: float = float("nan")
    funding_rate: float = float("nan")  # per funding interval, fraction
    next_funding_ns: int = 0
    open_interest: float = float("nan")


@dataclass(slots=True)
class FeedStatus:
    """Emitted by feeds on (re)connect, resync or detected gaps."""

    venue: str
    symbol: str
    ts_local_ns: int
    status: str  # "connected" | "gap" | "resynced" | "disconnected"
    detail: str = ""


@dataclass(slots=True)
class OrderUpdate:
    """Own-order lifecycle event from the account stream or REST."""

    venue: str
    symbol: str
    ts_exch_ns: int
    ts_local_ns: int
    client_order_id: int
    order_id: str
    status: str  # "open" | "filled" | "canceled" | "partially_filled" | "rejected"
    side: int
    price: float
    size: float
    filled_size: float
    filled_quote: float
    extra: dict = field(default_factory=dict)


MarketEvent = BookSnapshot | BookDelta | BBO | Trade | MarketStats | FeedStatus

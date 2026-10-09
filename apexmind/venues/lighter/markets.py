"""Lighter market metadata, integer encoding and order-size feasibility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

# Margin fractions are expressed in 1/10_000 units (the SDK computes
# ``imf = int(10_000 / leverage)``).
MARGIN_FRACTION_SCALE = 10_000


def _fee(v: Any, unit: str) -> float:
    x = float(v or 0.0)
    return x / 100.0 if unit == "percent" else x


@dataclass
class LighterMarket:
    market_id: int
    symbol: str
    status: str
    taker_fee: float  # fraction of notional
    maker_fee: float
    min_base_amount: float
    min_quote_amount: float
    size_decimals: int
    price_decimals: int
    default_imf: float  # fraction, e.g. 0.05 => 20x max default leverage
    min_imf: float  # smallest initial margin fraction allowed (max leverage)
    mmf: float  # maintenance margin fraction
    last_trade_price: float = math.nan
    daily_quote_volume: float = 0.0
    open_interest: float = 0.0
    market_type: str = "perp"

    @property
    def tick(self) -> float:
        return 10.0 ** -self.price_decimals

    @property
    def lot(self) -> float:
        return 10.0 ** -self.size_decimals

    @property
    def max_leverage(self) -> float:
        return 1.0 / self.min_imf if self.min_imf > 0 else 1.0

    @property
    def active(self) -> bool:
        return self.status.lower() == "active"

    # -- integer wire encoding ------------------------------------------------
    def price_to_int(self, px: float, side_is_buy: bool | None = None) -> int:
        """Encode a price; buys round down and sells round up when a side is
        given so a limit is never more aggressive than requested."""
        d = Decimal(str(px)) * (Decimal(10) ** self.price_decimals)
        if side_is_buy is True:
            return int(d.to_integral_value(rounding=ROUND_FLOOR))
        if side_is_buy is False:
            return int(d.to_integral_value(rounding=ROUND_CEILING))
        return int(d.to_integral_value())

    def size_to_int(self, qty: float) -> int:
        d = Decimal(str(qty)) * (Decimal(10) ** self.size_decimals)
        return int(d.to_integral_value(rounding=ROUND_FLOOR))

    def int_to_price(self, v: int) -> float:
        return v / 10.0**self.price_decimals

    def int_to_size(self, v: int) -> float:
        return v / 10.0**self.size_decimals

    def quantize_size(self, qty: float) -> float:
        return self.int_to_size(self.size_to_int(qty))

    def min_feasible_qty(self, price: float) -> float:
        """Smallest order size satisfying both base and quote minimums."""
        q = max(self.min_base_amount, self.min_quote_amount / price if price > 0 else math.inf)
        lots = math.ceil(q / self.lot - 1e-9)
        return lots * self.lot

    def order_is_feasible(self, qty: float, price: float) -> bool:
        return qty + 1e-12 >= self.min_base_amount and qty * price + 1e-9 >= self.min_quote_amount

    @classmethod
    def from_detail(cls, d: dict, fee_unit: str = "percent") -> "LighterMarket":
        def frac(key: str, default: float) -> float:
            v = d.get(key)
            return float(v) / MARGIN_FRACTION_SCALE if v not in (None, "") else default

        return cls(
            market_id=int(d["market_id"]),
            symbol=str(d["symbol"]),
            status=str(d.get("status", "active")),
            taker_fee=_fee(d.get("taker_fee"), fee_unit),
            maker_fee=_fee(d.get("maker_fee"), fee_unit),
            min_base_amount=float(d.get("min_base_amount") or 0.0),
            min_quote_amount=float(d.get("min_quote_amount") or 0.0),
            size_decimals=int(d.get("size_decimals", d.get("supported_size_decimals", 4))),
            price_decimals=int(d.get("price_decimals", d.get("supported_price_decimals", 2))),
            default_imf=frac("default_initial_margin_fraction", 0.1),
            min_imf=frac("min_initial_margin_fraction", 0.1),
            mmf=frac("maintenance_margin_fraction", 0.05),
            last_trade_price=float(d.get("last_trade_price") or math.nan),
            daily_quote_volume=float(d.get("daily_quote_token_volume") or 0.0),
            open_interest=float(d.get("open_interest") or 0.0),
            market_type=str(d.get("market_type", "perp")),
        )


def parse_order_book_details(payload: dict, fee_unit: str = "percent") -> dict[int, LighterMarket]:
    """Parse ``/api/v1/orderBookDetails``; spot markets are ignored."""
    rows = payload.get("order_book_details") or []
    out = {}
    for d in rows:
        m = LighterMarket.from_detail(d, fee_unit)
        if m.market_type.lower() in ("perp", "perps", "perpetual"):
            out[m.market_id] = m
    return out

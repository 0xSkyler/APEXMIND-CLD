"""Lighter <-> reference-venue instrument mapping.

A mapping is accepted only when the two venues' prices agree after an
explicit contract multiplier, so that e.g. a ``1000PEPE`` contract can never
be silently paired with a ``PEPE`` contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from apexmind.config import InstrumentConfig
from apexmind.venues.lighter.markets import LighterMarket

_MULTIPLIERS = (1.0, 1e3, 1e-3, 1e4, 1e-4, 1e6, 1e-6)


@dataclass
class Instrument:
    symbol: str  # canonical symbol == Lighter symbol
    market_id: int
    ref_venue: str
    ref_symbol: str
    price_multiplier: float  # lighter_price ~= ref_price * price_multiplier
    eligible: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def candidate_ref_symbols(symbol: str, quote: str = "USDT") -> list[str]:
    s = symbol.upper()
    out = [f"{s}{quote}", f"1000{s}{quote}", f"1000000{s}{quote}"]
    for prefix in ("1000000", "1000", "K"):
        if s.startswith(prefix) and len(s) > len(prefix):
            out.append(f"{s[len(prefix):]}{quote}")
    return list(dict.fromkeys(out))


def build_instruments(
    markets: dict[int, LighterMarket],
    ref_prices: dict[str, float],
    cfg: InstrumentConfig,
    ref_venue: str,
) -> list[Instrument]:
    out = []
    wanted = {b.upper() for b in cfg.initial_bases}
    for m in sorted(markets.values(), key=lambda x: x.market_id):
        cands = [cfg.overrides[m.symbol]] if m.symbol in cfg.overrides else candidate_ref_symbols(m.symbol)
        chosen, mult, why = "", math.nan, "no reference symbol"
        lp = m.last_trade_price
        for c in cands:
            rp = ref_prices.get(c)
            if rp is None or not rp > 0:
                continue
            if not (lp > 0):
                why = "no lighter price"
                break
            for k in _MULTIPLIERS:
                if abs(lp / (rp * k) - 1.0) <= cfg.max_price_ratio_deviation:
                    chosen, mult, why = c, k, "ok"
                    break
            if chosen:
                break
            why = f"price ratio mismatch vs {c}"
        eligible = bool(chosen) and m.active
        reason = why
        if chosen and not m.active:
            reason = f"market status {m.status}"
        elif chosen and m.daily_quote_volume < cfg.min_lighter_daily_quote_volume:
            eligible, reason = False, "insufficient lighter volume"
        elif chosen and not cfg.expand_to_all_eligible and m.symbol.upper() not in wanted:
            eligible, reason = False, "not in initial coverage (expansion disabled)"
        out.append(Instrument(m.symbol, m.market_id, ref_venue, chosen, mult, eligible, reason))
    return out

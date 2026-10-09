"""Synthetic two-venue market generator for METHODOLOGY VALIDATION ONLY.

It writes recordings in the collector's raw format (Lighter and Binance
messages plus ``meta`` records), so the complete research pipeline can be
exercised against data whose ground truth is known:

* positive control - Lighter's mid adjusts to the reference price with a lag
  (``lag_s`` > 0): a real, exploitable cross-venue signal exists;
* negative control - no lag (``lag_s`` = 0): any "alpha" the pipeline reports
  is an artefact of the methodology.

Every session record is stamped ``synthetic: true`` and every downstream
report built from it is labelled as synthetic. Nothing produced from this
module is evidence about real markets.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from apexmind.collector.recorder import RawRecorder
from apexmind.core.clock import ManualClock


@dataclass
class SymbolSpec:
    symbol: str = "ETH"
    market_id: int = 0
    price: float = 2000.0
    tick: float = 0.05
    size_decimals: int = 4
    price_decimals: int = 2
    vol_bps_sqrt_s: float = 3.0  # efficient-price volatility
    lighter_spread_ticks: int = 2
    ref_spread_ticks: int = 1
    level_size: float = 2.0  # mean base size per level
    lighter_trade_rate: float = 2.0  # per second
    ref_trade_rate: float = 8.0
    funding_rate: float = 1e-5  # per hour (fraction)


@dataclass
class SyntheticSpec:
    symbols: list[SymbolSpec] = field(default_factory=lambda: [SymbolSpec()])
    hours: float = 2.0
    start_ns: int = 1_767_225_600 * 10**9  # 2026-01-01T00:00:00Z
    dt_ms: int = 100
    lag_s: float = 3.0  # Lighter follows the reference with this delay
    kappa: float = 0.5  # per-step partial adjustment toward the lagged price
    lighter_noise_bps: float = 0.3
    informed_flow: float = 0.3  # bias of Lighter taker side toward the pending move
    regime_hours: float = 1.0  # volatility regime switch period
    resync_minutes: float = 30.0  # fresh book snapshots, like the collector
    high_vol_mult: float = 2.0
    taker_fee_pct: str = "0.0200"  # Lighter metadata convention: percent
    maker_fee_pct: str = "0.0020"
    lighter_delay_ms: tuple[float, float] = (40.0, 10.0)  # feed delay mean, sd
    ref_delay_ms: tuple[float, float] = (25.0, 5.0)
    seed: int = 1


def _fmt(x: float, d: int) -> str:
    return f"{x:.{d}f}"


class _Book:
    """Ten-level book around a mid; emits Lighter/Binance style diffs."""

    def __init__(self, tick: float, spread_ticks: int, level_size: float, rng: np.random.Generator, n: int = 10):
        self.tick, self.spread, self.size, self.rng, self.n = tick, spread_ticks, level_size, rng, n
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}

    def target(self, mid: float, tilt: float) -> tuple[dict[int, float], dict[int, float]]:
        b0 = math.floor(mid / self.tick - self.spread / 2.0)
        a0 = b0 + self.spread
        bids, asks = {}, {}
        for i in range(self.n):
            kb, ka = b0 - i, a0 + i
            bids[kb] = self.bids.get(kb) or round(self.size * self.rng.uniform(0.5, 1.5), 4)
            asks[ka] = self.asks.get(ka) or round(self.size * self.rng.uniform(0.5, 1.5), 4)
        # top-of-book sizes lean toward the side of the pending move
        bids[b0] = round(self.size * max(0.05, 1.0 + tilt) * self.rng.uniform(0.7, 1.3), 4)
        asks[a0] = round(self.size * max(0.05, 1.0 - tilt) * self.rng.uniform(0.7, 1.3), 4)
        return bids, asks

    def update(self, mid: float, tilt: float):
        nb, na = self.target(mid, tilt)
        db = [(k, 0.0) for k in self.bids if k not in nb] + [(k, v) for k, v in nb.items() if self.bids.get(k) != v]
        da = [(k, 0.0) for k in self.asks if k not in na] + [(k, v) for k, v in na.items() if self.asks.get(k) != v]
        self.bids, self.asks = nb, na
        return db, da

    def best(self) -> tuple[float, float, float, float]:
        kb, ka = max(self.bids), min(self.asks)
        return kb * self.tick, self.bids[kb], ka * self.tick, self.asks[ka]


def generate(spec: SyntheticSpec, data_dir: str) -> dict:
    """Write a synthetic recording; returns the ground-truth description."""
    rng = np.random.default_rng(spec.seed)
    clock = ManualClock(spec.start_ns)
    rec = RawRecorder(data_dir, clock, session_id=f"synthetic{spec.seed}")
    dt = spec.dt_ms / 1000.0
    steps = int(spec.hours * 3600 / dt)
    lag_steps = int(round(spec.lag_s / dt))
    truth = {"synthetic": True, "lag_s": spec.lag_s, "kappa": spec.kappa, "spec": asdict(spec)}
    details = []
    for s in spec.symbols:
        details.append({
            "symbol": s.symbol, "market_id": s.market_id, "market_type": "perp", "status": "active",
            "taker_fee": spec.taker_fee_pct, "maker_fee": spec.maker_fee_pct, "min_base_amount": "0.0010",
            "min_quote_amount": "10", "size_decimals": s.size_decimals, "price_decimals": s.price_decimals,
            "default_initial_margin_fraction": 500, "min_initial_margin_fraction": 500,
            "maintenance_margin_fraction": 300, "closeout_margin_fraction": 200, "last_trade_price": s.price,
            "daily_quote_token_volume": 1e9, "open_interest": 1e6,
        })
    t0 = spec.start_ns
    rec.write_meta("session", {"synthetic": True, "synthetic_truth": truth}, t0)
    rec.write_meta("lighter_markets", {"order_book_details": details}, t0)
    rec.write_meta("instruments", [
        {"symbol": s.symbol, "market_id": s.market_id, "ref_venue": "binance_usdm", "ref_symbol": f"{s.symbol}USDT",
         "price_multiplier": 1.0, "eligible": True, "reason": "synthetic"} for s in spec.symbols], t0)

    out: list[tuple[int, str, str, int, int]] = []  # (arrival, venue, raw, send time, seq)
    cur_t = [0]

    def emit(item: tuple[int, str, str]) -> None:
        """Queue a message; send time is the current step, seq its order."""
        out.append((item[0], item[1], item[2], cur_t[0], len(out)))

    def ld() -> int:
        return int(max(5.0, rng.normal(*spec.lighter_delay_ms)) * 1e6)

    def rd() -> int:
        return int(max(2.0, rng.normal(*spec.ref_delay_ms)) * 1e6)

    n_sym = len(spec.symbols)
    # correlated efficient-price shocks across symbols
    corr = np.full((n_sym, n_sym), 0.6) + 0.4 * np.eye(n_sym)
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((steps, n_sym)) @ chol.T
    regime = ((np.arange(steps) * dt / 3600.0 / spec.regime_hours).astype(int) % 2).astype(float)
    vol_mult = 1.0 + (spec.high_vol_mult - 1.0) * regime
    for j, s in enumerate(spec.symbols):
        sig = s.vol_bps_sqrt_s * 1e-4 * math.sqrt(dt)
        logp = math.log(s.price) + np.cumsum(shocks[:, j] * sig * vol_mult)
        lbook = _Book(s.tick, s.lighter_spread_ticks, s.level_size, rng)
        rbook = _Book(s.tick, s.ref_spread_ticks, s.level_size * 4, rng, n=20)
        l_log = logp[0]
        nonce = 1
        u_id = 1000
        ref_sym = f"{s.symbol}USDT"
        rs = ref_sym.lower()
        trade_id = 0
        last_bbo = None
        pd_ = s.price_decimals
        sd_ = s.size_decimals
        for i in range(steps):
            t = t0 + int((i + 1) * dt * 1e9)
            cur_t[0] = t
            ms_ = t // 1_000_000
            target = logp[i - lag_steps] if i >= lag_steps else logp[0]
            l_log = l_log + spec.kappa * (target - l_log) if lag_steps > 0 else logp[i]
            l_mid = math.exp(l_log + rng.normal(0, spec.lighter_noise_bps * 1e-4))
            r_mid = math.exp(logp[i])
            pending = (target - l_log) / (s.vol_bps_sqrt_s * 1e-4) if lag_steps > 0 else 0.0
            tilt = float(np.clip(0.3 * pending, -0.8, 0.8))
            # --- Lighter book ---
            resync = i > 0 and i % int(spec.resync_minutes * 60 / dt) == 0
            if resync:
                lbook.update(l_mid, tilt)
                b = [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in sorted(lbook.bids.items(), reverse=True)]
                a = [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in sorted(lbook.asks.items())]
                nonce += 1
                emit((t + ld(), "lighter", json.dumps({"type": "subscribed/order_book",
                            "channel": f"order_book:{s.market_id}", "timestamp": ms_,
                            "order_book": {"bids": b, "asks": a, "nonce": nonce}})))
                snap = {"lastUpdateId": u_id, "E": ms_, "T": ms_,
                        "bids": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in sorted(rbook.bids.items(), reverse=True)],
                        "asks": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in sorted(rbook.asks.items())]}
                emit((t + rd() + 1_000_000, "binance_usdm",
                            json.dumps({"stream": f"rest:depth:{ref_sym}", "data": snap})))
            if i == 0:
                lbook.update(l_mid, tilt)
                b = [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in sorted(lbook.bids.items(), reverse=True)]
                a = [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in sorted(lbook.asks.items())]
                raw = json.dumps({"type": "subscribed/order_book", "channel": f"order_book:{s.market_id}",
                                  "timestamp": ms_, "order_book": {"bids": b, "asks": a, "nonce": nonce}})
                emit((t + ld(), "lighter", raw))
                rbook.update(r_mid, 0.0)
                snap = {"lastUpdateId": u_id, "E": ms_, "T": ms_,
                        "bids": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in sorted(rbook.bids.items(), reverse=True)],
                        "asks": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in sorted(rbook.asks.items())]}
                emit((t + rd() + 50_000_000, "binance_usdm",
                            json.dumps({"stream": f"rest:depth:{ref_sym}", "data": snap})))
                continue
            db, da = lbook.update(l_mid, tilt)
            if db or da:
                prev = nonce
                nonce += 1 + int(rng.integers(0, 3))
                raw = json.dumps({"type": "update/order_book", "channel": f"order_book:{s.market_id}", "timestamp": ms_,
                                  "order_book": {"bids": [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in db],
                                                 "asks": [{"price": _fmt(k * s.tick, pd_), "size": _fmt(v, sd_)} for k, v in da],
                                                 "nonce": nonce, "begin_nonce": prev}})
                emit((t + ld(), "lighter", raw))
            # --- reference book (diff depth every step, BBO on change) ---
            rdb, rda = rbook.update(r_mid, 0.0)
            if rdb or rda:
                pu, U = u_id, u_id + 1
                u_id += 1 + int(rng.integers(0, 4))
                emit((t + rd(), "binance_usdm", json.dumps({"stream": f"{rs}@depth@100ms", "data": {
                    "e": "depthUpdate", "E": ms_, "T": ms_, "s": ref_sym, "U": U, "u": u_id, "pu": pu,
                    "b": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in rdb],
                    "a": [[_fmt(k * s.tick, pd_), _fmt(v, sd_)] for k, v in rda]}})))
            bbo = rbook.best()
            if bbo != last_bbo:
                last_bbo = bbo
                emit((t + rd(), "binance_usdm", json.dumps({"stream": f"{rs}@bookTicker", "data": {
                    "e": "bookTicker", "u": u_id, "E": ms_, "T": ms_, "s": ref_sym, "b": _fmt(bbo[0], pd_),
                    "B": _fmt(bbo[1], sd_), "a": _fmt(bbo[2], pd_), "A": _fmt(bbo[3], sd_)}})))
            # --- trades ---
            for _ in range(rng.poisson(s.ref_trade_rate * dt)):
                trade_id += 1
                up = shocks[i, j] > 0
                buy = rng.random() < (0.65 if up else 0.35)
                px = bbo[2] if buy else bbo[0]
                emit((t + rd(), "binance_usdm", json.dumps({"stream": f"{rs}@aggTrade", "data": {
                    "e": "aggTrade", "E": ms_, "a": trade_id, "s": ref_sym, "p": _fmt(px, pd_),
                    "q": _fmt(rng.exponential(s.level_size * 0.3), sd_), "T": ms_, "m": not buy}})))
            lb = lbook.best()
            for _ in range(rng.poisson(s.lighter_trade_rate * dt)):
                trade_id += 1
                p_buy = 0.5 + float(np.clip(spec.informed_flow * pending, -0.4, 0.4))
                buy = rng.random() < p_buy
                px = lb[2] if buy else lb[0]
                emit((t + ld(), "lighter", json.dumps({"type": "update/trade", "channel": f"trade:{s.market_id}",
                    "trades": [{"trade_id": trade_id, "market_id": s.market_id, "price": _fmt(px, pd_),
                                "size": _fmt(rng.exponential(s.level_size * 0.2), sd_), "is_maker_ask": buy,
                                "timestamp": ms_, "bid_account_id": 1, "ask_account_id": 2}]})))
            # --- funding / mark (1 Hz) ---
            if i % int(1 / dt) == 0:
                next_f = (t // 3_600_000_000_000 + 1) * 3_600_000_000_000
                emit((t + ld(), "lighter", json.dumps({"type": "update/market_stats",
                    "channel": f"market_stats:{s.market_id}", "timestamp": ms_, "market_stats": {
                        "market_id": s.market_id, "mark_price": _fmt(l_mid, pd_), "index_price": _fmt(r_mid, pd_),
                        "current_funding_rate": f"{s.funding_rate * 100:.6f}", "funding_timestamp": next_f // 1_000_000,
                        "open_interest": "1000"}})))
                emit((t + rd(), "binance_usdm", json.dumps({"stream": f"{rs}@markPrice@1s", "data": {
                    "e": "markPriceUpdate", "E": ms_, "s": ref_sym, "p": _fmt(r_mid, pd_), "i": _fmt(r_mid, pd_),
                    "r": f"{s.funding_rate:.8f}", "T": next_f // 1_000_000}})))
    # A venue's messages travel over one ordered connection: latency varies
    # but cannot reorder them. Sort by send order, then make arrival times
    # monotone per venue.
    ordered = []
    for venue in ("lighter", "binance_usdm"):
        rows = sorted((r for r in out if r[1] == venue), key=lambda r: (r[3], r[4]))
        last = 0
        for t_arr, v, raw, _t_send, _seq in rows:
            last = max(t_arr, last + 1_000)
            ordered.append((last, v, raw))
    ordered.sort(key=lambda r: r[0])
    out = ordered
    for t, venue, raw in out:
        rec.write(venue, "synthetic", t, raw)
    rec.close()
    truth["messages"] = len(out)
    return truth

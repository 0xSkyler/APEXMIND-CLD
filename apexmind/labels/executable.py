"""Execution-conditioned forecast targets.

For each valid decision at local time ``t`` and each side:

1. draw an execution delay ``L1`` from the latency model (one draw shared by
   both sides, so long/short targets use common random numbers);
2. at ``t + L1`` walk the *then-current* Lighter book for the label notional
   (aggressive entry) and, optionally, post at the touch (passive entry);
3. for each horizon ``H`` close at ``t + L1 + H + L2`` by walking the book
   again with the entered quantity;
4. subtract fees on both legs and funding if a funding time is crossed.

Actions are processed by the driver in local-time order interleaved with the
recorded events, so every simulated fill sees exactly the book that existed
at that instant on our timeline. Maximum adverse excursion is computed after
the replay from the per-tick executable marks.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from apexmind.config import LabelConfig
from apexmind.core.events import BUY, SELL, Trade
from apexmind.features.state import MarketState
from apexmind.latency.model import LatencyModel

SIDES = (("long", BUY), ("short", SELL))
_ENTRY, _EXIT, _PEXIT = 0, 1, 2


def label_columns(cfg: LabelConfig) -> list[str]:
    cols = ["t_entry_ns", "lat_entry_ms"]
    for s, _ in SIDES:
        cols += [f"entry_px_{s}", f"qty_{s}", f"slip_bps_{s}"]
        if cfg.passive:
            cols += [f"pfill_{s}", f"pwait_ms_{s}", f"ppx_{s}"]
        for h in cfg.horizons_s:
            cols += [f"ret_{s}_{h:g}", f"gross_{s}_{h:g}", f"fund_{s}_{h:g}", f"mae_{s}_{h:g}"]
            if cfg.passive:
                cols += [f"pret_{s}_{h:g}"]
    for h in cfg.horizons_s:
        cols += [f"hold_ms_{h:g}"]
    return cols


class LabelEngine:
    def __init__(self, symbols: list[str], cfg: LabelConfig, latency: LatencyModel,
                 fees: Callable[[str], tuple[float, float]], seed: int | None = None) -> None:
        self.cfg = cfg
        self.lat = latency
        self.fees = fees  # symbol -> (taker, maker)
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.cols = {c: [] for c in label_columns(cfg)}
        self.n = 0
        self._heap: list = []
        self._seq = 0
        # per-symbol executable marks on the grid, for MAE
        self.tick_bid: dict[str, list[float]] = {s: [] for s in symbols}
        self.tick_ask: dict[str, list[float]] = {s: [] for s in symbols}
        self._mae_spans: list[tuple[int, str, str, float, int, int, str]] = []
        self._passive: dict[str, list[list]] = defaultdict(list)
        self._lat_buf = np.empty(0)
        self._lat_pos = 0

    # -- helpers ----------------------------------------------------------------
    def _lat_ns(self) -> int:
        if self._lat_pos >= len(self._lat_buf):
            self._lat_buf = self.lat.sample_ms(self.rng, 4096)
            self._lat_pos = 0
        v = self._lat_buf[self._lat_pos]
        self._lat_pos += 1
        return int(v * 1e6)

    def _push(self, due: int, kind: int, payload: tuple) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (due, self._seq, kind, payload))

    def _set(self, col: str, row: int, v: float) -> None:
        self.cols[col][row] = v

    def next_due(self) -> int:
        return self._heap[0][0] if self._heap else 2**63 - 1

    # -- driver hooks -------------------------------------------------------------
    def add_row(self) -> int:
        for v in self.cols.values():
            v.append(math.nan)
        self.n += 1
        return self.n - 1

    def on_tick(self, t_ns: int, sym: str, bid: float, ask: float) -> None:
        self.tick_bid[sym].append(bid)
        self.tick_ask[sym].append(ask)
        lst = self._passive.get(sym)
        if lst:
            keep = []
            for tr in lst:
                if tr[5] < t_ns:
                    self._set(f"pfill_{tr[1]}", tr[0], 0.0)
                else:
                    keep.append(tr)
            self._passive[sym] = keep

    def schedule(self, row: int, t_ns: int, sym: str, mid: float) -> None:
        lat = self._lat_ns()
        self._set("lat_entry_ms", row, lat / 1e6)
        self._push(t_ns + lat, _ENTRY, (row, sym, mid))

    def run_next(self, ms: MarketState) -> None:
        due, _, kind, payload = heapq.heappop(self._heap)
        if kind == _ENTRY:
            self._entry(due, ms, *payload)
        elif kind == _EXIT:
            self._exit(due, ms, *payload)
        else:
            self._pexit(due, ms, *payload)

    def on_trade(self, tr: Trade) -> None:
        lst = self._passive.get(tr.symbol)
        if not lst:
            return
        keep = []
        for p in lst:
            row, side_name, side, px, queue, deadline, qty, f0 = p
            if tr.ts_local_ns > deadline:
                self._set(f"pfill_{side_name}", row, 0.0)
                continue
            # our resting buy is hit by sell takers (and vice versa)
            if tr.taker_side != -side:
                keep.append(p)
                continue
            through = tr.price < px * (1 - 1e-12) if side == BUY else tr.price > px * (1 + 1e-12)
            at = abs(tr.price - px) <= px * 1e-12
            if at:
                p[4] = queue - tr.size
            if through or (at and p[4] <= -qty):
                self._passive_fill(row, side_name, side, px, qty, tr.ts_local_ns, f0, tr.symbol)
            else:
                keep.append(p)
        self._passive[tr.symbol] = keep

    # -- actions ------------------------------------------------------------------
    def _entry(self, t: int, ms: MarketState, row: int, sym: str, mid: float) -> None:
        lv = ms.lit[sym]
        self._set("t_entry_ns", row, t)
        if not lv.book.usable():
            return
        n = self.cfg.notional_usd
        taker, _ = self.fees(sym)
        f0 = lv.next_funding_ns
        tick_idx = len(self.tick_bid[sym])
        top = lv.top()
        entries = {}
        for name, side in SIDES:
            f = lv.book.walk(side, notional=n)
            if not f.complete:
                continue
            self._set(f"entry_px_{name}", row, f.vwap)
            self._set(f"qty_{name}", row, f.qty)
            self._set(f"slip_bps_{name}", row, (f.vwap / mid - 1.0) * 1e4 * side)
            # executable mark immediately after entry (closing side of the touch)
            mark = (top[0] if side == BUY else top[2]) if top else math.nan
            entries[name] = (side, f.vwap, f.qty, mark)
        if entries:
            for h in self.cfg.horizons_s:
                self._push(t + int(h * 1e9) + self._lat_ns(), _EXIT, (row, sym, h, t, tick_idx, f0, entries, taker))
        if self.cfg.passive:
            top = lv.top()
            if top is not None:
                for name, side in SIDES:
                    px, queue = (top[0], top[1]) if side == BUY else (top[2], top[3])
                    self._set(f"ppx_{name}", row, px)
                    deadline = t + int(self.cfg.passive_wait_s * 1e9)
                    self._passive[sym].append([row, name, side, px, queue, deadline, n / px, f0])

    def _funding(self, ms: MarketState, sym: str, f0: int, t_in: int, t_out: int) -> float:
        """Funding paid by a long per unit notional over (t_in, t_out]."""
        lv = ms.lit[sym]
        if f0 and t_in < f0 <= t_out and not math.isnan(lv.funding_rate):
            return lv.funding_rate
        return 0.0

    def _exit(self, t: int, ms: MarketState, row: int, sym: str, h: float, t_in: int, tick_in: int, f0: int,
              entries: dict, taker: float) -> None:
        lv = ms.lit[sym]
        self._set(f"hold_ms_{h:g}", row, (t - t_in) / 1e6)
        if not lv.book.usable():
            return
        fund_long = self._funding(ms, sym, f0, t_in, t)
        tick_out = len(self.tick_bid[sym])
        for name, (side, px_in, qty, mark_in) in entries.items():
            f = lv.book.walk(-side, qty=qty)
            if not f.complete:
                continue
            px_out = f.vwap
            gross = (px_out - px_in) / px_in * side
            fund = -fund_long * side
            net = gross - taker * (px_in + px_out) / px_in + fund
            self._set(f"gross_{name}_{h:g}", row, gross)
            self._set(f"fund_{name}_{h:g}", row, fund)
            self._set(f"ret_{name}_{h:g}", row, net)
            self._mae_spans.append((row, name, sym, px_in, tick_in, tick_out, f"mae_{name}_{h:g}", mark_in, px_out))

    def _passive_fill(self, row: int, name: str, side: int, px: float, qty: float, t_fill: int, f0: int,
                      sym: str) -> None:
        self._set(f"pfill_{name}", row, 1.0)
        t_entry = self.cols["t_entry_ns"][row]
        self._set(f"pwait_ms_{name}", row, (t_fill - t_entry) / 1e6)
        for h in self.cfg.horizons_s:
            self._push(t_fill + int(h * 1e9) + self._lat_ns(), _PEXIT, (row, name, side, px, qty, t_fill, f0, h, sym))

    def _pexit(self, t: int, ms: MarketState, row: int, name: str, side: int, px_in: float, qty: float,
               t_in: int, f0: int, h: float, sym: str) -> None:
        lv = ms.lit[sym]
        if not lv.book.usable():
            return
        f = lv.book.walk(-side, qty=qty)
        if not f.complete:
            return
        taker, maker = self.fees(sym)
        fund = -self._funding(ms, sym, f0, t_in, t) * side
        net = (f.vwap - px_in) / px_in * side - (maker * px_in + taker * f.vwap) / px_in + fund
        self._set(f"pret_{name}_{h:g}", row, net)

    # -- finalize -------------------------------------------------------------------
    def finalize(self) -> dict[str, np.ndarray]:
        arrays = {c: np.asarray(v, dtype=float) for c, v in self.cols.items()}
        by_sym: dict[str, list] = defaultdict(list)
        for span in self._mae_spans:
            by_sym[span[2]].append(span)
        for sym, spans in by_sym.items():
            bids = np.asarray(self.tick_bid[sym], dtype=float)
            asks = np.asarray(self.tick_ask[sym], dtype=float)
            bmin = _RangeMin(np.where(np.isnan(bids), np.inf, bids))
            amax = _RangeMin(-np.where(np.isnan(asks), -np.inf, asks))
            for row, name, _, px_in, i0, i1, col, mark_in, px_out in spans:
                # worst executable mark: right after entry, on every grid tick
                # while open, and the exit fill itself
                if name == "long":
                    worst = min(mark_in, px_out, bmin.query(i0, i1) if i1 > i0 else math.inf)
                    arrays[col][row] = min(0.0, worst / px_in - 1.0)
                else:
                    worst = max(mark_in, px_out, -amax.query(i0, i1) if i1 > i0 else -math.inf)
                    arrays[col][row] = min(0.0, px_in / worst - 1.0)
        return arrays


class _RangeMin:
    """Sparse table for O(1) range-minimum queries over [i, j)."""

    def __init__(self, a: np.ndarray) -> None:
        self.levels = [a]
        k = 1
        while 2 * k <= len(a):
            prev = self.levels[-1]
            self.levels.append(np.minimum(prev[:-k], prev[k:]))
            k *= 2

    def query(self, i: int, j: int) -> float:
        n = j - i
        lvl = n.bit_length() - 1
        a = self.levels[lvl]
        return float(min(a[i], a[j - (1 << lvl)]))

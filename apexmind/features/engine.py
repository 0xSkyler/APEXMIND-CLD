"""Point-in-time feature engine.

``FeatureEngine.sample(t)`` reads :class:`MarketState` *as of* local time
``t`` (the caller guarantees every event with ``ts_local_ns <= t`` and no
later event has been applied) and returns one feature row per symbol. The
same code runs in research replay and in live trading, so there is no
batch/online skew. Windowed quantities are differences of cumulative
counters stored on the decision grid.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from apexmind.config import FeatureConfig
from apexmind.core.events import BUY, SELL
from apexmind.features.state import MarketState, VenueState

EPS = 1e-12


def _hl_alpha(halflife_s: float, dt_s: float) -> float:
    return 1.0 - 0.5 ** (dt_s / max(halflife_s, 1e-9))


@dataclass
class _Tick:
    l_mid: float
    r_mid: float
    l_buy: float
    l_sell: float
    l_trades: int
    r_buy: float
    r_sell: float
    r_trades: int
    l_ofi: float
    r_ofi: float
    l_add: float
    l_rem: float


class _Ewma:
    __slots__ = ("alpha", "mean", "var", "n")

    def __init__(self, alpha: float) -> None:
        self.alpha, self.mean, self.var, self.n = alpha, math.nan, math.nan, 0

    def update(self, x: float) -> None:
        if math.isnan(x):
            return
        if self.n == 0:
            self.mean, self.var = x, 0.0
        else:
            d = x - self.mean
            self.mean += self.alpha * d
            self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)
        self.n += 1

    def z(self, x: float) -> float:
        if self.n < 20 or not self.var > 0:
            return math.nan
        return (x - self.mean) / math.sqrt(self.var)


class _SymFeatures:
    def __init__(self, cfg: FeatureConfig, max_lag: int) -> None:
        dt = cfg.grid_ms / 1000.0
        self.hist: deque[_Tick] = deque(maxlen=max_lag + 1)
        self.basis = _Ewma(_hl_alpha(cfg.basis_ewma_halflife_s, dt))
        self.rv_l = _Ewma(_hl_alpha(cfg.vol_ewma_halflife_s, dt))
        self.rv_r = _Ewma(_hl_alpha(cfg.vol_ewma_halflife_s, dt))
        self.rv_long = _Ewma(_hl_alpha(cfg.regime_lookback_s, dt))
        self.spread = _Ewma(_hl_alpha(cfg.regime_lookback_s, dt))
        self.l1_depth = _Ewma(_hl_alpha(60.0, dt))
        self.r1_depth = _Ewma(_hl_alpha(60.0, dt))
        self.band_depth = _Ewma(_hl_alpha(60.0, dt))
        self.trade_rate = _Ewma(_hl_alpha(cfg.regime_lookback_s, dt))


class FeatureEngine:
    def __init__(self, symbols: list[str], cfg: FeatureConfig, exec_latency_ms=None) -> None:
        self.cfg = cfg
        self.symbols = list(symbols)
        self.dt_s = cfg.grid_ms / 1000.0
        self.ret_lags = [self._lag(h) for h in cfg.return_horizons_s]
        self.flow_lags = [self._lag(h) for h in cfg.flow_horizons_s]
        max_lag = max(self.ret_lags + self.flow_lags)
        self.state = {s: _SymFeatures(cfg, max_lag) for s in symbols}
        # callable(t_ns) -> current execution-latency estimate (ms)
        self.exec_latency_ms = exec_latency_ms or (lambda t: math.nan)
        self.names = self._names()

    def _lag(self, h: float) -> int:
        k = h / self.dt_s
        if abs(k - round(k)) > 1e-9 or round(k) < 1:
            raise ValueError(f"horizon {h}s is not a positive multiple of the {self.cfg.grid_ms}ms grid")
        return int(round(k))

    def _names(self) -> list[str]:
        n = []
        for h in self.cfg.return_horizons_s:
            n += [f"ref_ret_{h:g}", f"lit_ret_{h:g}", f"xret_{h:g}"]
        n += ["basis_bps", "basis_z"]
        n += ["lit_obi_l1", "lit_obi_depth", "ref_obi_l1", "ref_obi_depth"]
        for h in self.cfg.flow_horizons_s:
            n += [f"lit_ofi_{h:g}", f"ref_ofi_{h:g}", f"lit_aggr_{h:g}", f"ref_aggr_{h:g}",
                  f"lit_trade_rate_{h:g}", f"lit_cancel_{h:g}", f"lit_replenish_{h:g}"]
        n += ["lit_spread_bps", "ref_spread_bps", "lit_depth_bid_log", "lit_depth_ask_log",
              "lit_impact_buy_bps", "lit_impact_sell_bps", "lit_rv_bps", "ref_rv_bps",
              "lit_funding", "ref_funding", "funding_diff", "lit_funding_eta_s",
              "vol_ratio", "trend_60", "spread_ratio", "trade_rate_ratio",
              "lit_fresh_ms", "ref_fresh_ms", "fresh_diff_ms", "lit_delay_ms", "ref_delay_ms", "exec_latency_ms"]
        return n

    @staticmethod
    def _obi(vs: VenueState, n: int) -> tuple[float, float]:
        top = vs.top()
        if top is None:
            return math.nan, math.nan
        l1 = (top[1] - top[3]) / (top[1] + top[3] + EPS)
        if not vs.book.valid:
            return l1, math.nan
        b, a = vs.book.depth_qty(BUY, n), vs.book.depth_qty(SELL, n)
        return l1, (b - a) / (b + a + EPS)

    def sample(self, t_ns: int, ms: MarketState) -> dict[str, tuple[list[float], dict]]:
        """Return ``{symbol: (feature_values, info)}`` as of ``t_ns``.

        ``info`` carries non-feature context: validity, mids, touch prices.
        """
        cfg = self.cfg
        out = {}
        for sym in self.symbols:
            lv, rv, st = ms.lit[sym], ms.ref[sym], self.state[sym]
            l_mid, r_mid = lv.mid(), rv.mid()
            tick = _Tick(l_mid, r_mid, lv.buy_vol, lv.sell_vol, lv.trades, rv.buy_vol, rv.sell_vol, rv.trades,
                         lv.ofi, rv.ofi, lv.added, lv.removed)
            prev = st.hist[-1] if st.hist else None
            st.hist.append(tick)
            h = st.hist
            # -- EWMAs updated once per grid tick ---------------------------------
            if prev is not None:
                if l_mid > 0 and prev.l_mid > 0:
                    st.rv_l.update(math.log(l_mid / prev.l_mid) ** 2)
                if r_mid > 0 and prev.r_mid > 0:
                    r2 = math.log(r_mid / prev.r_mid) ** 2
                    st.rv_r.update(r2)
                    st.rv_long.update(r2)
                st.trade_rate.update(lv.trades - prev.l_trades)
            basis = math.log(l_mid / r_mid) * 1e4 if l_mid > 0 and r_mid > 0 else math.nan
            basis_z = st.basis.z(basis)
            st.basis.update(basis)
            ltop, rtop = lv.top(), rv.top()
            l_spread = (ltop[2] - ltop[0]) / l_mid * 1e4 if ltop and l_mid > 0 else math.nan
            r_spread = (rtop[2] - rtop[0]) / r_mid * 1e4 if rtop and r_mid > 0 else math.nan
            spread_ratio = l_spread / st.spread.mean if st.spread.n > 20 and st.spread.mean > 0 else math.nan
            st.spread.update(l_spread)
            if ltop:
                st.l1_depth.update(0.5 * (ltop[1] + ltop[3]))
            if rtop:
                st.r1_depth.update(0.5 * (rtop[1] + rtop[3]))
            depth_bid = lv.book.depth_notional_within(BUY, cfg.depth_band_bps) if lv.book.valid else math.nan
            depth_ask = lv.book.depth_notional_within(SELL, cfg.depth_band_bps) if lv.book.valid else math.nan
            if l_mid > 0 and not math.isnan(depth_bid):
                st.band_depth.update(0.5 * (depth_bid + depth_ask) / l_mid)

            vals: list[float] = []
            for k in self.ret_lags:
                if len(h) > k:
                    old = h[-1 - k]
                    rr = math.log(r_mid / old.r_mid) * 1e4 if r_mid > 0 and old.r_mid > 0 else math.nan
                    lr = math.log(l_mid / old.l_mid) * 1e4 if l_mid > 0 and old.l_mid > 0 else math.nan
                else:
                    rr = lr = math.nan
                vals += [rr, lr, rr - lr]
            vals += [basis, basis_z]
            l_obi1, l_obid = self._obi(lv, cfg.depth_levels)
            r_obi1, r_obid = self._obi(rv, cfg.depth_levels)
            vals += [l_obi1, l_obid, r_obi1, r_obid]
            l1d = st.l1_depth.mean if st.l1_depth.n else math.nan
            r1d = st.r1_depth.mean if st.r1_depth.n else math.nan
            bd = st.band_depth.mean if st.band_depth.n else math.nan
            for k in self.flow_lags:
                if len(h) > k:
                    old = h[-1 - k]
                    lofi = (tick.l_ofi - old.l_ofi) / (l1d + EPS)
                    rofi = (tick.r_ofi - old.r_ofi) / (r1d + EPS)
                    lb, ls = tick.l_buy - old.l_buy, tick.l_sell - old.l_sell
                    rb, rs = tick.r_buy - old.r_buy, tick.r_sell - old.r_sell
                    laggr = (lb - ls) / (lb + ls) if lb + ls > 0 else 0.0
                    raggr = (rb - rs) / (rb + rs) if rb + rs > 0 else 0.0
                    rate = (tick.l_trades - old.l_trades) / (k * self.dt_s)
                    added, removed = tick.l_add - old.l_add, tick.l_rem - old.l_rem
                    cancel = max(0.0, removed - (lb + ls)) / (bd + EPS)
                    repl = added / (removed + EPS) if removed > 0 else math.nan
                else:
                    lofi = rofi = laggr = raggr = rate = cancel = repl = math.nan
                vals += [lofi, rofi, laggr, raggr, rate, cancel, repl]
            imp_b = lv.book.impact_bps(BUY, cfg.impact_notional_usd) if lv.book.usable() else math.nan
            imp_s = lv.book.impact_bps(SELL, cfg.impact_notional_usd) if lv.book.usable() else math.nan
            l_rv = math.sqrt(st.rv_l.mean / self.dt_s) * 1e4 if st.rv_l.n > 1 else math.nan
            r_rv = math.sqrt(st.rv_r.mean / self.dt_s) * 1e4 if st.rv_r.n > 1 else math.nan
            r_rv_long = math.sqrt(st.rv_long.mean / self.dt_s) * 1e4 if st.rv_long.n > 1 else math.nan
            eta = (lv.next_funding_ns - t_ns) / 1e9 if lv.next_funding_ns else math.nan
            vol_ratio = r_rv / r_rv_long if r_rv_long and r_rv_long > 0 else math.nan
            k60 = self._lag(60.0) if 60.0 in cfg.return_horizons_s else None
            trend = math.nan
            if k60 is not None and len(h) > k60 and r_rv and r_rv > 0:
                old = h[-1 - k60]
                if r_mid > 0 and old.r_mid > 0:
                    trend = math.log(r_mid / old.r_mid) * 1e4 / (r_rv * math.sqrt(60.0))
            tr_ratio = math.nan
            if st.trade_rate.n > 20 and st.trade_rate.mean > 0 and len(h) > self.flow_lags[0]:
                k = self.flow_lags[0]
                tr_ratio = (tick.l_trades - h[-1 - k].l_trades) / k / st.trade_rate.mean
            lf = (t_ns - lv.last_book_ns) / 1e6 if lv.last_book_ns else math.nan
            rf = (t_ns - rv.last_book_ns) / 1e6 if rv.last_book_ns else math.nan
            vals += [l_spread, r_spread,
                     math.log1p(depth_bid) if depth_bid == depth_bid else math.nan,
                     math.log1p(depth_ask) if depth_ask == depth_ask else math.nan,
                     imp_b, imp_s, l_rv, r_rv,
                     lv.funding_rate * 1e4, rv.funding_rate * 1e4, (lv.funding_rate - rv.funding_rate) * 1e4, eta,
                     vol_ratio, trend, spread_ratio, tr_ratio,
                     lf, rf, lf - rf, lv.delay_ms.quantile(0.5), rv.delay_ms.quantile(0.5),
                     float(self.exec_latency_ms(t_ns))]
            fresh_ok = (lf <= cfg.max_staleness_ms) and (rf <= cfg.max_staleness_ms)
            info = {
                "valid": bool(lv.usable() and rv.usable() and fresh_ok and len(h) > max(self.ret_lags)),
                "lit_usable": lv.usable(), "ref_usable": rv.usable(), "fresh_ok": bool(fresh_ok),
                "lit_mid": l_mid, "ref_mid": r_mid,
                "lit_bid": ltop[0] if ltop else math.nan, "lit_ask": ltop[2] if ltop else math.nan,
            }
            out[sym] = (vals, info)
        return out

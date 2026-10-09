"""Build point-in-time feature/label datasets from recorded data.

A single pass over the recording, in local-arrival order:

* events update :class:`MarketState`;
* on every decision-grid tick the :class:`FeatureEngine` samples features
  from the state *before* any later event is applied;
* the :class:`LabelEngine` executes simulated orders at their (latency-
  delayed) due times against the book that existed at that instant.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from apexmind import __version__
from apexmind.collector.quality import QualityMonitor
from apexmind.collector.recorder import iter_raw, list_files
from apexmind.config import Config
from apexmind.core.events import LIGHTER, Trade
from apexmind.data.replay import Replay
from apexmind.features.engine import FeatureEngine
from apexmind.features.state import MarketState
from apexmind.labels.executable import LabelEngine
from apexmind.latency.model import LatencyModel, build_latency_model
from apexmind.util import atomic_write_json, stable_hash

log = logging.getLogger(__name__)

FEATURE_VERSION = 1
INFO_COLS = ["valid", "lit_usable", "ref_usable", "fresh_ok", "lit_mid", "ref_mid", "lit_bid", "lit_ask"]


@dataclass
class DatasetMeta:
    start_ns: int
    end_ns: int
    symbols: list[str]
    feature_names: list[str]
    label_cols: list[str]
    latency: dict
    fees: dict
    quality: dict
    instruments: dict
    sessions: int
    data_fingerprint: str
    data_kind: str  # "recorded" | "synthetic"
    markets: dict = field(default_factory=dict)  # symbol -> LighterMarket fields (as recorded)
    rows: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def warmup_ns(cfg: Config) -> int:
    """Feature warm-up: longest window plus EWMA settling."""
    longest = max(cfg.features.return_horizons_s + cfg.features.flow_horizons_s)
    return int(max(2 * longest, 120.0) * 1e9)


def run_pipeline(events, symbols: list[str], ref_venue: str, cfg: Config, latency: LatencyModel, fees,
                 start_ns: int, end_ns: int, with_labels: bool = True, quality: QualityMonitor | None = None,
                 exec_latency_ms=None, label_tail_ns: int = 0, seed: int | None = None):
    """Drive state/features/labels over an event iterator. Returns
    ``(feature_names, rows_ts, rows_sym, feature_matrix, info, labels)``.

    Events before ``start_ns - warmup`` only update market state (book
    reconstruction from the last snapshot); during the warm-up window the
    feature engine also samples, and rows are emitted from ``start_ns``.
    """
    warm_start = start_ns - warmup_ns(cfg)
    ms = MarketState(symbols, ref_venue, cfg.features.depth_band_bps)
    fe = FeatureEngine(symbols, cfg.features, exec_latency_ms or (lambda t: latency.estimate_at(t)))
    le = LabelEngine(symbols, cfg.labels, latency, fees, seed) if with_labels else None
    grid = cfg.features.grid_ms * 1_000_000
    next_tick = None
    ts_list, sym_list, feat_rows = [], [], []
    info_rows: dict[str, list] = {c: [] for c in INFO_COLS}

    def do_tick(t: int) -> None:
        sampled = fe.sample(t, ms)
        if t < start_ns:
            return  # warm-up: rolling windows advance, no rows emitted
        for sym, (vals, info) in sampled.items():
            ts_list.append(t)
            sym_list.append(sym)
            feat_rows.append(vals)
            for c in INFO_COLS:
                info_rows[c].append(info[c])
            if le is not None:
                row = le.add_row()
                le.on_tick(t, sym, info["lit_bid"], info["lit_ask"])
                if info["valid"]:
                    le.schedule(row, t, sym, info["lit_mid"])

    def advance(t: int) -> None:
        nonlocal next_tick
        while True:
            due = le.next_due() if le is not None else 2**63 - 1
            if next_tick < t and next_tick <= due and next_tick < end_ns:
                do_tick(next_tick)
                next_tick += grid
            elif due < t:
                le.run_next(ms)
            else:
                break

    for ev in events:
        t = ev.ts_local_ns
        if t >= end_ns + label_tail_ns:
            break
        if t < warm_start:
            ms.apply(ev)
            continue
        if next_tick is None:
            next_tick = (t // grid + 1) * grid
        advance(t)
        ms.apply(ev)
        if quality is not None and start_ns <= t < end_ns:
            quality.observe(ev)
        if le is not None and isinstance(ev, Trade) and ev.venue == LIGHTER:
            le.on_trade(ev)
    if next_tick is not None:
        # drain: exits pending at the window end complete with data from the
        # label tail (no rows are emitted there)
        advance(end_ns + label_tail_ns)
    feats = np.asarray(feat_rows, dtype=np.float32).reshape(len(feat_rows), len(fe.names))
    labels = le.finalize() if le is not None else {}
    return fe.names, ts_list, sym_list, feats, info_rows, labels


def _fingerprint(files: list[tuple[str, Path]]) -> str:
    return stable_hash([(v, str(p.name), p.stat().st_size) for v, p in sorted(files)])


def scan_meta(data_dir, start_ns: int = 0, end_ns: int = 2**63 - 1) -> Replay:
    """Read only metadata records (markets, mappings, executions)."""
    rp = Replay(data_dir)
    for t, venue, _c, raw in iter_raw(data_dir, [], start_ns, end_ns):
        rp._meta(raw, t)
    return rp


def make_fee_fn(cfg: Config, rp: Replay):
    def fees(sym: str) -> tuple[float, float]:
        if cfg.lighter.fee_source == "config":
            return cfg.lighter.taker_fee, cfg.lighter.maker_fee
        for m in rp.markets.values():
            if m.symbol == sym:
                return m.taker_fee, m.maker_fee
        return cfg.lighter.taker_fee, cfg.lighter.maker_fee
    return fees


DAY_NS = 86_400 * 10**9


def _day_chunks(start_ns: int, end_ns: int) -> list[tuple[int, int]]:
    out, cs = [], start_ns
    while cs < end_ns:
        ce = min(end_ns, (cs // DAY_NS + 1) * DAY_NS)
        out.append((cs, ce))
        cs = ce
    return out


def label_tail_ns(cfg: Config, latency: LatencyModel) -> int:
    span_s = max(cfg.labels.horizons_s) + cfg.labels.passive_wait_s + 2 * latency.quantile(0.999) / 1e3 + 5.0
    return int(span_s * 1e9)


def _merge_quality(reports: list[dict]) -> dict:
    """Combine per-chunk quality reports: counts add, maxima take the max,
    delay quantiles are message-weighted averages (an approximation)."""
    out: dict[str, dict] = {}
    weight: dict[str, int] = {}
    for rep in reports:
        for k, q in rep.items():
            if k not in out:
                out[k], weight[k] = dict(q), q["messages"]
                continue
            o, w0, w1 = out[k], weight[k], q["messages"]
            for c in ("messages", "trades", "book_updates", "snapshots", "gaps", "disconnects", "exch_ts_regressions"):
                o[c] += q[c]
            o["max_interarrival_ms"] = max(o["max_interarrival_ms"], q["max_interarrival_ms"])
            o["span_hours"] = round(o["span_hours"] + q["span_hours"], 3)
            for c in ("delay_ms_p50", "delay_ms_p95", "delay_ms_p99"):
                if o[c] is not None and q[c] is not None and w0 + w1 > 0:
                    o[c] = round((o[c] * w0 + q[c] * w1) / (w0 + w1), 2)
            weight[k] = w0 + w1
    return out


def build_dataset(cfg: Config, data_dir: str, start_ns: int, end_ns: int, cache_dir: str | None = None,
                  latency: LatencyModel | None = None) -> tuple[pd.DataFrame, DatasetMeta]:
    """Build features + labels for ``[start_ns, end_ns)``.

    Work is split into UTC-day chunks cached independently: completed days
    never change, so a growing research window only rebuilds the newest day.
    Each chunk replays from its own snapshot lookback and completes labels
    from a short tail after its end, so chunking does not censor labels.
    """
    files = list_files(data_dir, ["meta", LIGHTER, cfg.reference.venue], start_ns, end_ns)
    if not files:
        raise FileNotFoundError(f"no recorded data in {data_dir} for the requested period")
    meta_rp = scan_meta(data_dir, 0, end_ns)
    # Detected, never configurable: synthetic data can't be labelled as recorded.
    data_kind = "synthetic" if any(x.get("synthetic") for x in meta_rp.sessions) else "recorded"
    symbols = sorted(s for s, i in meta_rp.instruments.items() if i.eligible)
    if not symbols:
        raise ValueError("recording contains no eligible instruments")
    if latency is None:
        latency = build_latency_model(meta_rp.exec_records, cfg.labels.latency_source,
                                      cfg.labels.prior_latency_median_ms, cfg.labels.prior_latency_sigma,
                                      cfg.labels.seed)
    fees = make_fee_fn(cfg, meta_rp)
    frames, quals, keys = [], [], []
    names: list[str] = []
    label_cols: list[str] = []
    for cs, ce in _day_chunks(start_ns, end_ns):
        df_c, qual_c, key, names, label_cols = _build_chunk(cfg, data_dir, cs, ce, symbols, latency, fees, cache_dir)
        frames.append(df_c)
        quals.append(qual_c)
        keys.append(key)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.attrs["feature_names"] = list(names)
    meta = DatasetMeta(
        start_ns=start_ns, end_ns=end_ns, symbols=symbols, feature_names=names, label_cols=label_cols,
        latency=latency.summary(), fees={s: fees(s) for s in symbols}, quality=_merge_quality(quals),
        instruments={s: i.to_dict() for s, i in meta_rp.instruments.items()}, sessions=len(meta_rp.sessions),
        data_fingerprint=_fingerprint(files), data_kind=data_kind, rows=len(df),
        markets={m.symbol: asdict(m) for m in meta_rp.markets.values() if m.symbol in symbols},
        extra={"chunk_cache_keys": keys, "synthetic_truth": meta_rp.sessions[0].get("synthetic_truth")
               if meta_rp.sessions else None},
    )
    return df, meta


def _build_chunk(cfg: Config, data_dir: str, cs: int, ce: int, symbols: list[str], latency: LatencyModel, fees,
                 cache_dir: str | None):
    # Books need their most recent snapshot: the collector refreshes them
    # every ``snapshot_resync_minutes``, which bounds the lookback.
    lookback = int((cfg.collector.snapshot_resync_minutes + 5) * 60e9) + warmup_ns(cfg)
    tail = label_tail_ns(cfg, latency)
    files = list_files(data_dir, ["meta", LIGHTER, cfg.reference.venue], max(0, cs - lookback), ce + tail)
    key = stable_hash({
        "fp": _fingerprint(files), "start": cs, "end": ce, "features": cfg.features.__dict__,
        "labels": cfg.labels.__dict__, "fees": {s: fees(s) for s in symbols}, "lat": latency.summary(),
        "symbols": symbols, "fv": FEATURE_VERSION, "v": __version__,
    })
    if cache_dir:
        cpath = Path(cache_dir) / f"chunk_{key}.parquet"
        qpath = Path(cache_dir) / f"chunk_{key}.json"
        if cpath.exists() and qpath.exists():
            log.info("dataset chunk cache hit %s", cpath.name)
            side = json.loads(qpath.read_text())
            return pd.read_parquet(cpath), side["quality"], key, side["names"], side["labels"]
    rp = Replay(data_dir, cfg.reference.venue)
    quality = QualityMonitor()
    seed = int(stable_hash([cfg.labels.seed, cs]), 16) % 2**32  # independent latency draws per chunk
    names, ts, syms, feats, info, labels = run_pipeline(
        rp.events(max(0, cs - lookback), ce + tail), symbols, cfg.reference.venue, cfg, latency, fees,
        cs, ce, quality=quality, label_tail_ns=tail, seed=seed)
    df = assemble_frame(names, ts, syms, feats, info, labels)
    q = quality.report()
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
        atomic_write_json(qpath, {"quality": q, "names": list(names), "labels": list(labels)})
    return df, q, key, list(names), list(labels)


def assemble_frame(names, ts, syms, feats, info, labels) -> pd.DataFrame:
    data = {"ts_ns": np.asarray(ts, dtype=np.int64), "symbol": pd.Categorical(syms)}
    for c in INFO_COLS:
        arr = np.asarray(info[c])
        data[c] = arr.astype(bool) if c in ("valid", "lit_usable", "ref_usable", "fresh_ok") else arr.astype(float)
    for j, n in enumerate(names):
        data[n] = feats[:, j]
    for c, arr in labels.items():
        data[c] = arr
    df = pd.DataFrame(data)
    df.attrs["feature_names"] = list(names)
    return df


def summarize_period(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "start_utc": pd.Timestamp(int(df.ts_ns.min()), unit="ns", tz="UTC").isoformat(),
        "end_utc": pd.Timestamp(int(df.ts_ns.max()), unit="ns", tz="UTC").isoformat(),
        "hours": round((df.ts_ns.max() - df.ts_ns.min()) / 3.6e12, 3),
        "rows": int(len(df)),
        "valid_fraction": float(df.valid.mean()),
        "symbols": sorted(map(str, df.symbol.unique())),
    }


def nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x

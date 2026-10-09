"""Execution-latency model.

The quantity that matters for research is the delay, on the *local*
timeline, between a decision and the moment our order's fill is visible on
the public trade feed. Using that definition aligns simulated fills with the
recorded book timeline regardless of exchange clock offsets.

Measured samples come from live execution records (``meta``/``exec``). Until
enough exist, a conservative lognormal prior is used and every report says
so explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_MEASURED_SAMPLES = 30


@dataclass
class LatencyModel:
    samples_ms: np.ndarray
    source: str  # "measured" | "prior"
    detail: str = ""
    sample_times_ns: np.ndarray | None = None

    @classmethod
    def prior(cls, median_ms: float, sigma: float, n: int = 20_000, seed: int = 0) -> "LatencyModel":
        rng = np.random.default_rng(seed)
        s = rng.lognormal(math.log(median_ms), sigma, n)
        return cls(np.sort(s), "prior", f"lognormal(median={median_ms}ms, sigma={sigma})")

    @classmethod
    def from_exec_records(cls, records: list[dict], kind: str = "taker") -> "LatencyModel | None":
        """Build from execution records carrying ``t_decision`` and
        ``t_fill_print`` (local ns). Returns None if too few samples."""
        rows = [
            (r["t_decision"], (r["t_fill_print"] - r["t_decision"]) / 1e6)
            for r in records
            if r.get("kind", "taker") == kind and r.get("t_fill_print") and r.get("t_decision")
            and r["t_fill_print"] > r["t_decision"]
        ]
        if len(rows) < MIN_MEASURED_SAMPLES:
            return None
        rows.sort()
        t = np.array([r[0] for r in rows], dtype=np.int64)
        v = np.array([r[1] for r in rows], dtype=float)
        return cls(v, "measured", f"{len(v)} live {kind} executions", t)

    def __post_init__(self) -> None:
        self._median = float(np.median(self.samples_ms)) if len(self.samples_ms) else math.nan

    def sample_ms(self, rng: np.random.Generator, size: int | None = None):
        idx = rng.integers(0, len(self.samples_ms), size=size)
        return self.samples_ms[idx]

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples_ms, q))

    def stressed(self, q: float = 0.9) -> "LatencyModel":
        """Distribution shifted so its median equals the current q-quantile."""
        shift = self.quantile(q) / max(self.quantile(0.5), 1e-9)
        return LatencyModel(self.samples_ms * shift, self.source + f"+stress(p{int(q * 100)})", self.detail)

    def estimate_at(self, t_ns: int) -> float:
        """Median latency using only samples measured before ``t_ns``
        (point-in-time feature); falls back to the overall median."""
        if self.sample_times_ns is None:
            return self._median
        k = int(np.searchsorted(self.sample_times_ns, t_ns, side="right"))
        if k < MIN_MEASURED_SAMPLES:
            return float(np.median(self.samples_ms[: max(k, 1)])) if k else math.nan
        return float(np.median(self.samples_ms[max(0, k - 500) : k]))

    def summary(self) -> dict:
        return {
            "source": self.source,
            "detail": self.detail,
            "n": int(len(self.samples_ms)),
            "p50_ms": round(self.quantile(0.5), 1),
            "p90_ms": round(self.quantile(0.9), 1),
            "p99_ms": round(self.quantile(0.99), 1),
        }


def build_latency_model(exec_records: list[dict], source: str, prior_median_ms: float, prior_sigma: float,
                        seed: int = 0) -> LatencyModel:
    if source == "measured":
        m = LatencyModel.from_exec_records(exec_records)
        if m is not None:
            return m
    return LatencyModel.prior(prior_median_ms, prior_sigma, seed=seed)

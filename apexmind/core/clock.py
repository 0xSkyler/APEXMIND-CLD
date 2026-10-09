"""Local time base, exchange clock-offset estimation and feed-delay tracking.

All decisions are made on the *local arrival* timeline. ``Clock.now_ns`` is a
wall-clock anchor advanced by the monotonic clock, so an NTP step during a
session cannot reorder events. Exchange event timestamps are kept alongside,
and the offset between each exchange clock and ours is estimated NTP-style
from request round trips (offset = server - midpoint, error <= rtt / 2).
"""

from __future__ import annotations

import bisect
import math
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass


class Clock:
    def __init__(self) -> None:
        self._anchor_wall = time.time_ns()
        self._anchor_mono = time.monotonic_ns()

    def now_ns(self) -> int:
        return self._anchor_wall + (time.monotonic_ns() - self._anchor_mono)

    @staticmethod
    def wall_ns() -> int:
        return time.time_ns()

    @staticmethod
    def mono_ns() -> int:
        return time.monotonic_ns()

    def wall_drift_ns(self) -> int:
        """How far the system wall clock has moved away from our timeline."""
        return time.time_ns() - self.now_ns()

    def reanchor(self) -> int:
        """Re-anchor to the wall clock; returns the step applied (ns).

        Only call at file-rotation boundaries so that a single recording file
        is always on one consistent timeline.
        """
        before = self.now_ns()
        self._anchor_wall = time.time_ns()
        self._anchor_mono = time.monotonic_ns()
        return self._anchor_wall - before


class ManualClock(Clock):
    """Deterministic clock for replay and tests."""

    def __init__(self, start_ns: int = 0) -> None:
        self._t = start_ns

    def now_ns(self) -> int:
        return self._t

    def set(self, t_ns: int) -> None:
        if t_ns < self._t:
            raise ValueError("ManualClock cannot go backwards")
        self._t = t_ns

    def advance(self, dt_ns: int) -> None:
        self._t += dt_ns

    def wall_drift_ns(self) -> int:
        return 0

    def reanchor(self) -> int:
        return 0


@dataclass
class OffsetSample:
    t_local_ns: int
    offset_ns: float  # exchange_clock - local_clock
    rtt_ns: int


class OffsetEstimator:
    """NTP-style estimator of (exchange clock - local clock).

    Keeps a window of probes and reports the offset from the probe with the
    smallest round trip, whose error bound (rtt/2) is the tightest.
    """

    def __init__(self, window: int = 64) -> None:
        self.samples: deque[OffsetSample] = deque(maxlen=window)

    def add_probe(self, t_send_local_ns: int, t_server_ns: int, t_recv_local_ns: int) -> OffsetSample:
        rtt = t_recv_local_ns - t_send_local_ns
        if rtt < 0:
            raise ValueError("negative round trip")
        offset = t_server_ns - (t_send_local_ns + t_recv_local_ns) / 2.0
        s = OffsetSample(t_recv_local_ns, offset, rtt)
        self.samples.append(s)
        return s

    def best(self) -> OffsetSample | None:
        if not self.samples:
            return None
        return min(self.samples, key=lambda s: s.rtt_ns)

    @property
    def offset_ns(self) -> float:
        b = self.best()
        return b.offset_ns if b else 0.0

    @property
    def uncertainty_ns(self) -> float:
        b = self.best()
        return b.rtt_ns / 2.0 if b else math.inf


class RollingQuantiles:
    """Exact quantiles over a bounded window (sorted insert, O(window))."""

    def __init__(self, window: int = 2048) -> None:
        self.window = window
        self._fifo: deque[float] = deque()
        self._sorted: list[float] = []

    def add(self, x: float) -> None:
        self._fifo.append(x)
        bisect.insort(self._sorted, x)
        if len(self._fifo) > self.window:
            old = self._fifo.popleft()
            del self._sorted[bisect.bisect_left(self._sorted, old)]

    def __len__(self) -> int:
        return len(self._sorted)

    def quantile(self, q: float) -> float:
        if not self._sorted:
            return math.nan
        idx = min(len(self._sorted) - 1, max(0, int(round(q * (len(self._sorted) - 1)))))
        return self._sorted[idx]


class FeedDelayMonitor:
    """Tracks per-stream delay (local arrival - exchange time - clock offset)
    and staleness (time since last message) on the local timeline."""

    def __init__(self, window: int = 2048) -> None:
        self._delays: dict[str, RollingQuantiles] = {}
        self._last_local: dict[str, int] = {}
        self._window = window

    def observe(self, stream: str, ts_local_ns: int, ts_exch_ns: int | None, offset_ns: float = 0.0) -> None:
        self._last_local[stream] = ts_local_ns
        if ts_exch_ns:
            rq = self._delays.setdefault(stream, RollingQuantiles(self._window))
            # exchange time expressed on the local timeline = ts_exch - offset
            rq.add((ts_local_ns - (ts_exch_ns - offset_ns)) / 1e6)

    def delay_ms(self, stream: str, q: float = 0.5) -> float:
        rq = self._delays.get(stream)
        return rq.quantile(q) if rq else math.nan

    def staleness_ms(self, stream: str, now_ns: int) -> float:
        last = self._last_local.get(stream)
        return math.inf if last is None else (now_ns - last) / 1e6

    def streams(self) -> list[str]:
        return sorted(self._last_local)

    def snapshot(self, now_ns: int) -> dict[str, dict[str, float]]:
        return {
            s: {
                "staleness_ms": self.staleness_ms(s, now_ns),
                "delay_p50_ms": self.delay_ms(s, 0.5),
                "delay_p95_ms": self.delay_ms(s, 0.95),
            }
            for s in self.streams()
        }


_CHRONY_OFFSET = re.compile(r"System time\s*:\s*([0-9.]+) seconds (fast|slow)")
_CHRONY_ROOT = re.compile(r"Root dispersion\s*:\s*([0-9.]+) seconds")


def system_clock_status() -> dict[str, float | str]:
    """Best-effort read of the OS time-sync daemon (chrony) state.

    Returns offset/dispersion in milliseconds when chrony is available; the
    caller decides what to do when it is not (risk guard treats unknown sync
    as a reason to tighten, not loosen, limits).
    """
    if shutil.which("chronyc") is None:
        return {"source": "unavailable"}
    try:
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return {"source": "error"}
    res: dict[str, float | str] = {"source": "chrony"}
    m = _CHRONY_OFFSET.search(out)
    if m:
        sign = 1.0 if m.group(2) == "fast" else -1.0
        res["offset_ms"] = sign * float(m.group(1)) * 1e3
    m = _CHRONY_ROOT.search(out)
    if m:
        res["root_dispersion_ms"] = float(m.group(1)) * 1e3
    return res

"""Raw message recorder and time-ordered replay reader.

File layout::

    <data_dir>/<venue>/<YYYYMMDD>/<HH>.<session>.tsv.gz

Each line is ``ts_local_ns \\t ts_wall_ns \\t conn_id \\t raw_message``.
Raw venue messages are stored unmodified so parsing bugs can be fixed and
history re-derived. The special venue ``meta`` carries market metadata,
instrument mappings, clock probes and execution-latency measurements.
"""

from __future__ import annotations

import gzip
import heapq
import io
import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from apexmind.core.clock import Clock

META = "meta"


def _hour_key(ts_ns: int) -> tuple[str, str]:
    d = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return d.strftime("%Y%m%d"), d.strftime("%H")


class RawRecorder:
    def __init__(self, data_dir: str | os.PathLike, clock: Clock, session_id: str | None = None,
                 compresslevel: int = 3) -> None:
        self.root = Path(data_dir)
        self.clock = clock
        self.session = session_id or uuid.uuid4().hex[:12]
        self.compresslevel = compresslevel
        self._files: dict[tuple[str, str, str], io.TextIOBase] = {}
        self.lines = 0
        self.bytes = 0

    def _fh(self, venue: str, ts_local_ns: int):
        day, hour = _hour_key(ts_local_ns)
        key = (venue, day, hour)
        fh = self._files.get(key)
        if fh is None:
            # Close files for hours that have ended for this venue.
            for k in [k for k in self._files if k[0] == venue]:
                self._files.pop(k).close()
            path = self.root / venue / day / f"{hour}.{self.session}.tsv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = gzip.open(path, "at", compresslevel=self.compresslevel, encoding="utf-8")
            self._files[key] = fh
        return fh

    def write(self, venue: str, conn: str, ts_local_ns: int, raw: str) -> None:
        if "\n" in raw:
            raw = raw.replace("\n", " ")
        line = f"{ts_local_ns}\t{time.time_ns()}\t{conn}\t{raw}\n"
        self._fh(venue, ts_local_ns).write(line)
        self.lines += 1
        self.bytes += len(line)

    def write_meta(self, kind: str, payload, ts_local_ns: int | None = None) -> None:
        t = self.clock.now_ns() if ts_local_ns is None else ts_local_ns
        self.write(META, kind, t, json.dumps({"type": kind, "data": payload}, separators=(",", ":"), default=str))

    def flush(self) -> None:
        for fh in self._files.values():
            fh.flush()

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def _parse_lines(path: Path, venue: str, start_ns: int, end_ns: int) -> Iterator[tuple[int, str, str, str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 3)
                if len(parts) != 4:
                    continue  # truncated tail of a crashed session
                t = int(parts[0])
                if t < start_ns:
                    continue
                if t >= end_ns:
                    break
                yield t, venue, parts[2], parts[3]
    except (EOFError, gzip.BadGzipFile, OSError):
        # A process killed mid-write leaves a truncated gzip member; keep the
        # complete prefix and let the quality report flag the session.
        return


def list_files(data_dir: str | os.PathLike, venues: Iterable[str], start_ns: int, end_ns: int) -> list[tuple[str, Path]]:
    root = Path(data_dir)
    out = []
    lo = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    for venue in venues:
        vdir = root / venue
        if not vdir.exists():
            continue
        for day_dir in sorted(vdir.iterdir()):
            for f in sorted(day_dir.glob("*.tsv.gz")):
                hour = f.name.split(".", 1)[0]
                try:
                    t0 = datetime.strptime(day_dir.name + hour, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if t0 >= lo and t0.timestamp() * 1e9 < end_ns:
                    out.append((venue, f))
    return out


def iter_raw(data_dir: str | os.PathLike, venues: Iterable[str], start_ns: int = 0,
             end_ns: int = 2**63 - 1) -> Iterator[tuple[int, str, str, str]]:
    """Yield ``(ts_local_ns, venue, conn, raw)`` merged in local-time order.

    ``meta`` lines sort before market lines with the same timestamp so that
    metadata written at session start is applied first.
    """
    venues = list(dict.fromkeys([META, *venues]))
    streams = [_parse_lines(p, v, start_ns, end_ns) for v, p in list_files(data_dir, venues, start_ns, end_ns)]
    yield from heapq.merge(*streams, key=lambda r: (r[0], r[1] != META))


def data_span(data_dir: str | os.PathLike, venues: Iterable[str]) -> tuple[int, int] | None:
    """(first, last) local timestamps present for the given venues."""
    files = list_files(data_dir, venues, 0, 2**63 - 1)
    if not files:
        return None
    first, last = None, None
    for v, p in files:
        for t, *_ in _parse_lines(p, v, 0, 2**63 - 1):
            first = t if first is None else min(first, t)
            last = t if last is None else max(last, t)
    return (first, last) if first is not None else None

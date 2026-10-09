"""Helpers shared by venue adapters."""

from __future__ import annotations

import json
from typing import Any


def to_ns(ts: Any) -> int:
    """Convert an exchange timestamp of unknown unit (s/ms/us/ns) to ns.

    Units are inferred from magnitude, valid for dates between 2001 and 2286.
    """
    if ts is None or ts == "":
        return 0
    if isinstance(ts, str):
        ts = int(ts) if ts.lstrip("-").isdigit() else float(ts)
    if isinstance(ts, float):
        if ts <= 0:
            return 0
        if ts < 1e11:  # fractional seconds
            return int(round(ts * 1e9))
        ts = int(ts)
    v = int(ts)
    if v <= 0:
        return 0
    if v > 10**17:
        return v
    if v > 10**14:
        return v * 1_000
    if v > 10**11:
        return v * 1_000_000
    return v * 1_000_000_000


def levels(raw: list) -> list[tuple[float, float]]:
    """Parse ``[[px, sz], ...]`` or ``[{"price":..,"size":..}, ...]``."""
    out = []
    for lv in raw or ():
        if isinstance(lv, dict):
            out.append((float(lv["price"]), float(lv["size"])))
        else:
            out.append((float(lv[0]), float(lv[1])))
    return out


def loads(raw: str | bytes | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)

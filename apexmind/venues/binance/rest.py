"""Binance USD-M futures public REST endpoints."""

from __future__ import annotations

from apexmind.config import ReferenceConfig
from apexmind.core.clock import Clock
from apexmind.core.ratelimit import TokenBucket
from apexmind.venues.http import HttpClient, HttpResult


def depth_weight(limit: int) -> int:
    if limit <= 50:
        return 2
    if limit <= 100:
        return 5
    if limit <= 500:
        return 10
    return 20


class BinanceRest:
    def __init__(self, cfg: ReferenceConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.http = HttpClient(cfg.rest_url, TokenBucket.per_minute(cfg.rest_weight_per_minute), clock)

    async def close(self) -> None:
        await self.http.close()

    async def depth(self, symbol: str, limit: int | None = None) -> HttpResult:
        limit = limit or self.cfg.snapshot_depth
        return await self.http.get("/fapi/v1/depth", {"symbol": symbol, "limit": limit}, weight=depth_weight(limit))

    async def server_time_probe(self) -> tuple[int, int, int]:
        res = await self.http.get("/fapi/v1/time")
        return res.t_send_ns, int(res.data["serverTime"]) * 1_000_000, res.t_recv_ns

    async def prices(self) -> dict[str, float]:
        res = await self.http.get("/fapi/v1/ticker/price", weight=5)
        return {r["symbol"]: float(r["price"]) for r in res.data}

    async def exchange_info(self) -> HttpResult:
        return await self.http.get("/fapi/v1/exchangeInfo", weight=1)

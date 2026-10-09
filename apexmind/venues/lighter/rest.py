"""Lighter public/account REST endpoints used by the system.

Transaction signing and submission go through :mod:`apexmind.venues.lighter.trading`.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime

from apexmind.config import LighterConfig
from apexmind.core.clock import Clock
from apexmind.core.ratelimit import TokenBucket
from apexmind.venues.http import HttpClient, HttpResult
from apexmind.venues.lighter.markets import LighterMarket, parse_order_book_details


class LighterRest:
    def __init__(self, cfg: LighterConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.http = HttpClient(cfg.api_url, TokenBucket.per_minute(cfg.rest_requests_per_minute), clock)

    async def close(self) -> None:
        await self.http.close()

    async def order_book_details(self, market_id: int | None = None) -> HttpResult:
        params = {"market_id": market_id} if market_id is not None else None
        return await self.http.get("/api/v1/orderBookDetails", params)

    async def markets(self, fee_unit: str = "percent") -> dict[int, LighterMarket]:
        res = await self.order_book_details()
        return parse_order_book_details(res.data, fee_unit)

    async def recent_trades(self, market_id: int, limit: int = 100) -> HttpResult:
        return await self.http.get("/api/v1/recentTrades", {"market_id": market_id, "limit": limit})

    async def funding_rates(self) -> HttpResult:
        return await self.http.get("/api/v1/funding-rates")

    async def fundings(self, market_id: int, start_s: int, end_s: int, resolution: str = "1h") -> HttpResult:
        return await self.http.get(
            "/api/v1/fundings",
            {"market_id": market_id, "resolution": resolution, "start_timestamp": start_s,
             "end_timestamp": end_s, "count_back": 0},
        )

    async def exchange_stats(self) -> HttpResult:
        return await self.http.get("/api/v1/exchangeStats")

    async def account(self, account_index: int) -> HttpResult:
        return await self.http.get("/api/v1/account", {"by": "index", "value": str(account_index)})

    async def active_orders(self, account_index: int, market_id: int, auth: str) -> HttpResult:
        return await self.http.get(
            "/api/v1/accountActiveOrders",
            {"account_index": account_index, "market_id": market_id},
            headers={"Authorization": auth},
        )

    async def inactive_orders(self, account_index: int, auth: str, limit: int = 100) -> HttpResult:
        return await self.http.get(
            "/api/v1/accountInactiveOrders",
            {"account_index": account_index, "limit": limit},
            headers={"Authorization": auth},
        )

    async def account_limits(self, account_index: int, auth: str) -> HttpResult:
        return await self.http.get("/api/v1/accountLimits", {"account_index": account_index},
                                   headers={"Authorization": auth})

    async def next_nonce(self, account_index: int, api_key_index: int) -> int:
        res = await self.http.get("/api/v1/nextNonce", {"account_index": account_index, "api_key_index": api_key_index})
        return int(res.data["nonce"])

    async def position_funding(self, account_index: int, auth: str, limit: int = 100) -> HttpResult:
        return await self.http.get("/api/v1/positionFunding", {"account_index": account_index, "limit": limit},
                                   headers={"Authorization": auth})

    async def server_time_probe(self) -> tuple[int, int, int] | None:
        """(t_send_local, t_server, t_recv_local) from the HTTP Date header.

        The Date header has 1 s resolution, so this only bounds gross clock
        errors; fine-grained alignment relies on local-arrival timestamps.
        """
        res = await self.http.get("/api/v1/exchangeStats")
        date = res.headers.get("Date") or res.headers.get("date")
        if not date:
            return None
        t_server = int(parsedate_to_datetime(date).timestamp() * 1e9) + 500_000_000
        return res.t_send_ns, t_server, res.t_recv_ns

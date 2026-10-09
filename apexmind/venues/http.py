"""Small async HTTP helper with rate limiting, retries and timing capture."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from apexmind.core.clock import Clock
from apexmind.core.ratelimit import TokenBucket

log = logging.getLogger(__name__)


@dataclass
class HttpResult:
    status: int
    data: Any
    t_send_ns: int
    t_recv_ns: int
    headers: dict[str, str]


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class HttpClient:
    def __init__(self, base_url: str, bucket: TokenBucket, clock: Clock, timeout_s: float = 10.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.bucket = bucket
        self.clock = clock
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.retries = retries
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # trust_env: honour HTTPS_PROXY / CA settings of the host.
            self._session = aiohttp.ClientSession(timeout=self.timeout, trust_env=True)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def request(self, method: str, path: str, *, params=None, data=None, headers=None, weight: float = 1.0) -> HttpResult:
        url = f"{self.base_url}{path}"
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            await self.bucket.acquire(weight)
            sess = await self.session()
            t0 = self.clock.now_ns()
            try:
                async with sess.request(method, url, params=params, data=data, headers=headers) as resp:
                    body = await resp.text()
                    t1 = self.clock.now_ns()
                    if resp.status == 429 or resp.status >= 500:
                        raise HttpError(resp.status, body)
                    if resp.status >= 400:
                        raise HttpError(resp.status, body)  # not retried below
                    try:
                        payload = await resp.json(content_type=None)
                    except Exception:
                        payload = body
                    return HttpResult(resp.status, payload, t0, t1, dict(resp.headers))
            except HttpError as e:
                last_exc = e
                if e.status < 500 and e.status != 429:
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
            if attempt < self.retries:
                log.warning("retrying %s %s after %s", method, path, last_exc)
                await asyncio.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

    async def get(self, path: str, params=None, weight: float = 1.0, headers=None) -> HttpResult:
        return await self.request("GET", path, params=params, weight=weight, headers=headers)

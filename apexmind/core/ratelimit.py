"""Weighted token-bucket rate limiting for exchange requests/transactions."""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: float | None = None, clock=time.monotonic) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate_per_s
        self.capacity = burst if burst is not None else max(1.0, rate_per_s)
        self.tokens = self.capacity
        self._clock = clock
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self, weight: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= weight:
            self.tokens -= weight
            return True
        return False

    def wait_time(self, weight: float = 1.0) -> float:
        self._refill()
        return max(0.0, (weight - self.tokens) / self.rate)

    async def acquire(self, weight: float = 1.0) -> None:
        if weight > self.capacity:
            raise ValueError("weight exceeds bucket capacity")
        async with self._lock:
            while not self.try_acquire(weight):
                await asyncio.sleep(self.wait_time(weight))

    @classmethod
    def per_minute(cls, n: float, burst_fraction: float = 0.1) -> "TokenBucket":
        return cls(n / 60.0, burst=max(1.0, n * burst_fraction))

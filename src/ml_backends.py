"""ML backend health checking with caching."""

import os
import time
from dataclasses import dataclass, field

import httpx


CACHE_TTL = 30  # seconds


@dataclass
class MLBackend:
    """A registered ML backend service."""

    name: str
    url: str
    icon: str
    _cache: dict = field(default_factory=dict, repr=False)

    async def fetch_health(self) -> dict | None:
        """Fetch /health from the backend. Returns the JSON dict or None on failure.

        Results are cached for ``CACHE_TTL`` seconds.
        """
        now = time.monotonic()
        if self._cache and now - self._cache.get("ts", 0) < CACHE_TTL:
            return self._cache.get("data")

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.url}/health")
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            data = None

        self._cache["data"] = data
        self._cache["ts"] = now
        return data


ML_BACKENDS: list[MLBackend] = [
    MLBackend(
        name="SAM 2",
        url=os.getenv("SAM2_ML_BACKEND_URL", "http://sam2-ml-backend:9090"),
        icon="memory",
    ),
]

"""Protection for a public deployment. All of it is set by environment variables,
and off by default, so local use stays simple:

- `API_KEY`: `X-API-Key` is required for the endpoints that change monitoring results.
  With `DATABASE_URL` set but no key, those endpoints refuse every request, so a
  persistent database is never left open by a missing setting.
- `RATE_LIMIT_PER_MINUTE`: requests per client that can reach the database. A valid key
  is exempt, because the Streamlit demo serves all of its visitors from one address.
- `MAX_BATCH_SIZE`: the most leads in one `POST /predict/batch`, for every caller.
- `VISITOR_RETENTION_DAYS`, `MAX_VISITOR_ROWS`: how much visitor data to keep.
- `TRUST_CF_CONNECTING_IP`: use Cloudflare's client address header. Set it only behind
  Cloudflare (as on Render), where Cloudflare sets the header; elsewhere a client could
  fake it. `X-Forwarded-For` is never used, because it keeps what the client sent.
"""

import math
import os
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from fastapi.security import APIKeyHeader

from apex.api.database import database_url

API_KEY_HEADER = "X-API-Key"
DEFAULT_MAX_BATCH_SIZE = 10_000
PROTECTED_PATHS = frozenset({"/outcomes", "/monitoring/run"})

# Shows the key in /docs ("Authorize"). The check itself runs earlier, in the middleware,
# so an unauthorized request cannot reach body parsing or the rejected-request log.
api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    key_required: bool = False
    rate_limit_per_minute: int | None = None
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    retention_days: int | None = None
    max_visitor_rows: int | None = None
    trust_cf_connecting_ip: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("API_KEY", "").strip() or None
        return cls(
            api_key=api_key,
            key_required=bool(api_key or database_url()),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE"),
            max_batch_size=_env_int("MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE),
            retention_days=_env_int("VISITOR_RETENTION_DAYS"),
            max_visitor_rows=_env_int("MAX_VISITOR_ROWS"),
            trust_cf_connecting_ip=os.environ.get("TRUST_CF_CONNECTING_IP", "").strip().lower()
            == "true",
        )


def has_valid_key(request: Request, settings: Settings) -> bool:
    given = request.headers.get(API_KEY_HEADER)
    if not settings.api_key or given is None:
        return False
    return secrets.compare_digest(given.encode(), settings.api_key.encode())


def client_address(request: Request, settings: Settings) -> str:
    if settings.trust_cf_connecting_ip:
        address = request.headers.get("CF-Connecting-IP")
        if address:
            return address
    return request.client.host if request.client else "unknown"


def is_rate_limited_path(request: Request) -> bool:
    """Requests that can write to, or wake, the database. /health and /docs stay free."""
    path = request.url.path
    return request.method == "POST" or path == "/health/db" or path.startswith("/monitoring")


class RateLimiter:
    """Sliding window per client, kept in memory: one count per API worker."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ):
        self.limit, self.window, self.clock, self.max_clients = (
            limit,
            window_seconds,
            clock,
            max_clients,
        )
        self.hits: dict[str, deque[float]] = {}

    def allow(self, client: str) -> bool:
        now = self.clock()
        hits = self.hits.setdefault(client, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        if len(self.hits) > self.max_clients:
            self._forget_idle_clients(now)
        return True

    def retry_after(self, client: str) -> int:
        """Whole seconds until `client` can send again."""
        hits = self.hits.get(client)
        if not hits:
            return 0
        return max(1, math.ceil(hits[0] + self.window - self.clock()))

    def _forget_idle_clients(self, now: float) -> None:
        self.hits = {c: h for c, h in self.hits.items() if h and h[-1] > now - self.window}
        while len(self.hits) > self.max_clients:
            del self.hits[next(iter(self.hits))]

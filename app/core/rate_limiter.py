"""Persistent, per-IP rate limiting for credit-consuming HTTP endpoints.

The limiter uses the same writable runtime database directory as authentication,
so limits are shared by all workers that can see that filesystem. Client IPs are
hashed before storage and raw addresses are never written to disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import aiosqlite
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.runtime_paths import get_runtime_directory


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting without making startup fragile."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RateLimitWindow:
    limit: int
    seconds: int


@dataclass(frozen=True)
class RateLimitRule:
    name: str
    method: str
    path: str
    windows: tuple[RateLimitWindow, ...]


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int = 0


class SQLiteIPRateLimiter:
    """Atomic sliding-window limiter backed by SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create storage once, including on platforms that skip ASGI lifespan."""
        if self._initialized:
            return

        async with self._initialize_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=5000")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rate_limit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bucket TEXT NOT NULL,
                        identity_hash TEXT NOT NULL,
                        occurred_at REAL NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_lookup
                    ON rate_limit_events (bucket, identity_hash, occurred_at)
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_cleanup
                    ON rate_limit_events (bucket, occurred_at)
                    """
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS active_monitor_leases (
                        monitor_id TEXT PRIMARY KEY,
                        identity_hash TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_monitor_lease_owner
                    ON active_monitor_leases (identity_hash, expires_at)
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_monitor_lease_cleanup
                    ON active_monitor_leases (expires_at)
                    """
                )
                await db.commit()
            self._initialized = True

    async def check_and_record(
        self,
        bucket: str,
        identity_hash: str,
        windows: Sequence[RateLimitWindow],
    ) -> RateLimitDecision:
        await self.initialize()
        now = time.time()
        longest_window = max(window.seconds for window in windows)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "DELETE FROM rate_limit_events WHERE bucket = ? AND occurred_at < ?",
                    (bucket, now - longest_window),
                )

                checks: list[tuple[RateLimitWindow, int, float | None]] = []
                for window in windows:
                    cutoff = now - window.seconds
                    async with db.execute(
                        """
                        SELECT COUNT(*), MIN(occurred_at)
                        FROM rate_limit_events
                        WHERE bucket = ? AND identity_hash = ? AND occurred_at > ?
                        """,
                        (bucket, identity_hash, cutoff),
                    ) as cursor:
                        count, oldest = await cursor.fetchone()
                    checks.append((window, int(count), oldest))

                blocked = [check for check in checks if check[1] >= check[0].limit]
                if blocked:
                    window, _, oldest = max(
                        blocked,
                        key=lambda check: (check[2] or now) + check[0].seconds - now,
                    )
                    reset_at = int((oldest or now) + window.seconds) + 1
                    await db.rollback()
                    return RateLimitDecision(
                        allowed=False,
                        limit=window.limit,
                        remaining=0,
                        reset_at=reset_at,
                        retry_after=max(1, reset_at - int(now)),
                    )

                await db.execute(
                    """
                    INSERT INTO rate_limit_events (bucket, identity_hash, occurred_at)
                    VALUES (?, ?, ?)
                    """,
                    (bucket, identity_hash, now),
                )
                await db.commit()

                tightest = min(
                    checks,
                    key=lambda check: (check[0].limit - check[1] - 1) / check[0].limit,
                )
                window, count, oldest = tightest
                reset_at = int((oldest or now) + window.seconds) + 1
                return RateLimitDecision(
                    allowed=True,
                    limit=window.limit,
                    remaining=max(0, window.limit - count - 1),
                    reset_at=reset_at,
                )
            except Exception:
                await db.rollback()
                raise

    async def acquire_monitor_slot(
        self,
        identity_hash: str,
        limit: int,
        expires_at: float,
    ) -> str | None:
        """Atomically reserve a monitor slot shared by all database workers."""
        await self.initialize()
        now = time.time()
        monitor_id = str(uuid.uuid4())[:8]

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "DELETE FROM active_monitor_leases WHERE expires_at <= ?",
                    (now,),
                )
                async with db.execute(
                    """
                    SELECT COUNT(*)
                    FROM active_monitor_leases
                    WHERE identity_hash = ? AND expires_at > ?
                    """,
                    (identity_hash, now),
                ) as cursor:
                    active_count = int((await cursor.fetchone())[0])

                if active_count >= max(1, limit):
                    await db.rollback()
                    return None

                await db.execute(
                    """
                    INSERT INTO active_monitor_leases
                        (monitor_id, identity_hash, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (monitor_id, identity_hash, expires_at),
                )
                await db.commit()
                return monitor_id
            except Exception:
                await db.rollback()
                raise

    async def release_monitor_slot(self, monitor_id: str) -> None:
        """Release a persistent monitor lease when the task stops or expires."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute(
                "DELETE FROM active_monitor_leases WHERE monitor_id = ?",
                (monitor_id,),
            )
            await db.commit()


def _validated_ip(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate[1:candidate.index("]")]
    elif candidate.count(":") == 1 and "." in candidate:
        candidate = candidate.rsplit(":", 1)[0]
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        return None


def get_client_ip(request: Request, trust_proxy_headers: bool = False) -> str:
    """Resolve a client IP while avoiding spoofable proxy headers by default."""
    if trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        for entry in forwarded_for.split(","):
            parsed = _validated_ip(entry)
            if parsed:
                return parsed

        for header in ("x-real-ip", "x-vercel-forwarded-for"):
            parsed = _validated_ip(request.headers.get(header))
            if parsed:
                return parsed

    if request.client:
        parsed = _validated_ip(request.client.host)
        if parsed:
            return parsed
    return "unknown"


def get_rate_limit_pepper() -> str:
    """Return the required secret used to make stored IP hashes unlinkable."""
    pepper = os.getenv("NOVA_RATE_LIMIT_PEPPER", "").strip()
    if len(pepper) < 32:
        raise RuntimeError(
            "NOVA_RATE_LIMIT_PEPPER must be set to a private value of at least "
            "32 characters when rate limiting is enabled."
        )
    return pepper


def hash_client_ip(client_ip: str, pepper: str) -> str:
    return hmac.new(
        pepper.encode("utf-8"),
        client_ip.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def validate_rate_limit_configuration() -> None:
    """Fail at startup instead of silently hashing IPs with a public default."""
    if _env_bool("NOVA_RATE_LIMIT_ENABLED", True):
        get_rate_limit_pepper()


def build_rate_limit_rules() -> tuple[RateLimitRule, ...]:
    return (
        RateLimitRule(
            name="intelligence",
            method="POST",
            path="/api/command",
            windows=(
                RateLimitWindow(_env_int("NOVA_COMMANDS_PER_MINUTE", 10), 60),
                RateLimitWindow(_env_int("NOVA_COMMANDS_PER_DAY", 100), 86_400),
            ),
        ),
        RateLimitRule(
            name="monitor",
            method="POST",
            path="/api/monitor",
            windows=(
                RateLimitWindow(_env_int("NOVA_MONITORS_PER_HOUR", 2), 3_600),
                RateLimitWindow(_env_int("NOVA_MONITORS_PER_DAY", 4), 86_400),
            ),
        ),
        RateLimitRule(
            name="otp",
            method="POST",
            path="/api/auth/send-otp",
            windows=(
                RateLimitWindow(_env_int("NOVA_OTPS_PER_TEN_MINUTES", 5), 600),
                RateLimitWindow(_env_int("NOVA_OTPS_PER_DAY", 20), 86_400),
            ),
        ),
    )


def get_rate_limit_db_path() -> str:
    directory = get_runtime_directory(
        "database",
        env_var="NOVA_DB_DIR",
        local_directory=Path(__file__).resolve().parents[2],
    )
    return str(directory / "nova_rate_limits.db")


rate_limiter = SQLiteIPRateLimiter(get_rate_limit_db_path())


class IPRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply configured rules and expose standard quota headers."""

    def __init__(
        self,
        app,
        limiter: SQLiteIPRateLimiter,
        rules: Iterable[RateLimitRule],
    ) -> None:
        super().__init__(app)
        self.limiter = limiter
        self.rules = {
            (rule.method.upper(), rule.path.rstrip("/") or "/"): rule
            for rule in rules
        }
        self.enabled = _env_bool("NOVA_RATE_LIMIT_ENABLED", True)
        self.pepper = get_rate_limit_pepper() if self.enabled else secrets.token_urlsafe(32)
        self.trust_proxy_headers = _env_bool(
            "NOVA_TRUST_PROXY_HEADERS",
            default=bool(os.getenv("VERCEL")),
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        client_ip = get_client_ip(request, self.trust_proxy_headers)
        identity_hash = hash_client_ip(client_ip, self.pepper)
        request.state.client_ip_hash = identity_hash

        rule = self.rules.get(
            (request.method.upper(), request.url.path.rstrip("/") or "/")
        )
        if not self.enabled or rule is None:
            return await call_next(request)

        decision = await self.limiter.check_and_record(
            rule.name,
            identity_hash,
            rule.windows,
        )
        headers = {
            "X-RateLimit-Limit": str(decision.limit),
            "X-RateLimit-Remaining": str(decision.remaining),
            "X-RateLimit-Reset": str(decision.reset_at),
            "X-RateLimit-Policy": ", ".join(
                f"{window.limit};w={window.seconds}" for window in rule.windows
            ),
        }

        if not decision.allowed:
            headers["Retry-After"] = str(decision.retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "This IP address has reached its fair-use limit. "
                        f"Try again in {decision.retry_after} seconds."
                    ),
                    "code": "rate_limit_exceeded",
                    "retry_after": decision.retry_after,
                },
                headers=headers,
            )

        response = await call_next(request)
        response.headers.update(headers)
        return response

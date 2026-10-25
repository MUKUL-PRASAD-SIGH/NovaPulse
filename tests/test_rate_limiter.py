import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import httpx
from fastapi import FastAPI
from starlette.requests import Request

from app.api.ws import ConnectionManager, ContinuousIntelligenceEngine
from app.core.rate_limiter import (
    IPRateLimitMiddleware,
    RateLimitRule,
    RateLimitWindow,
    SQLiteIPRateLimiter,
    get_client_ip,
    get_rate_limit_pepper,
)


class RateLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env_patcher = patch.dict(
            os.environ,
            {"NOVA_RATE_LIMIT_PEPPER": "test-only-private-pepper-value-123456"},
        )
        self.env_patcher.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "limits.db")
        self.limiter = SQLiteIPRateLimiter(self.db_path)
        await self.limiter.initialize()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()
        self.env_patcher.stop()

    async def test_cleanup_and_monitor_lease_indexes_exist(self):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("PRAGMA index_list('rate_limit_events')") as cursor:
                rate_indexes = {row[1] for row in await cursor.fetchall()}
            async with db.execute("PRAGMA index_list('active_monitor_leases')") as cursor:
                monitor_indexes = {row[1] for row in await cursor.fetchall()}

        self.assertIn("idx_rate_limit_cleanup", rate_indexes)
        self.assertIn("idx_monitor_lease_cleanup", monitor_indexes)
        self.assertIn("idx_monitor_lease_owner", monitor_indexes)

    def test_ip_hash_pepper_is_required(self):
        with patch.dict(os.environ, {"NOVA_RATE_LIMIT_PEPPER": ""}):
            with self.assertRaisesRegex(RuntimeError, "at least 32 characters"):
                get_rate_limit_pepper()

    async def test_blocks_only_the_ip_that_reaches_its_limit(self):
        windows = (RateLimitWindow(limit=2, seconds=60),)

        first = await self.limiter.check_and_record("command", "ip-a", windows)
        second = await self.limiter.check_and_record("command", "ip-a", windows)
        blocked = await self.limiter.check_and_record("command", "ip-a", windows)
        other_ip = await self.limiter.check_and_record("command", "ip-b", windows)

        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 1)
        self.assertTrue(second.allowed)
        self.assertEqual(second.remaining, 0)
        self.assertFalse(blocked.allowed)
        self.assertGreaterEqual(blocked.retry_after, 1)
        self.assertTrue(other_ip.allowed)

    async def test_middleware_returns_quota_headers_and_json_429(self):
        app = FastAPI()
        app.add_middleware(
            IPRateLimitMiddleware,
            limiter=self.limiter,
            rules=(
                RateLimitRule(
                    name="test-command",
                    method="POST",
                    path="/api/command",
                    windows=(RateLimitWindow(limit=1, seconds=60),),
                ),
            ),
        )

        @app.post("/api/command")
        async def command():
            return {"ok": True}

        transport = httpx.ASGITransport(app=app, client=("198.51.100.20", 1234))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            allowed = await client.post("/api/command")
            blocked = await client.post("/api/command")

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["x-ratelimit-remaining"], "0")
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["code"], "rate_limit_exceeded")
        self.assertIn("retry-after", blocked.headers)

    async def test_monitor_slot_is_coordinated_across_limiter_instances(self):
        second_worker = SQLiteIPRateLimiter(self.db_path)
        expires_at = time.time() + 60

        first_id = await self.limiter.acquire_monitor_slot("same-ip", 1, expires_at)
        blocked_id = await second_worker.acquire_monitor_slot("same-ip", 1, expires_at)
        other_ip_id = await second_worker.acquire_monitor_slot("other-ip", 1, expires_at)

        self.assertIsNotNone(first_id)
        self.assertIsNone(blocked_id)
        self.assertIsNotNone(other_ip_id)

        await self.limiter.release_monitor_slot(first_id)
        retry_id = await second_worker.acquire_monitor_slot("same-ip", 1, expires_at)
        self.assertIsNotNone(retry_id)

        await second_worker.release_monitor_slot(other_ip_id)
        await second_worker.release_monitor_slot(retry_id)

    async def test_monitor_releases_slot_at_duration_deadline(self):
        engine = ContinuousIntelligenceEngine(ConnectionManager())
        released = []
        release_event = asyncio.Event()

        async def release_slot(monitor_id):
            released.append(monitor_id)
            release_event.set()

        with patch("app.memory.manager.memory_manager") as memory_manager:
            memory_manager.detect_changes.return_value = {"significant": False}
            monitor_id = await engine.start_monitor(
                topic="deadline test",
                interval_minutes=60,
                duration_hours=0.00001,
                monitor_id="deadline",
                release_callback=release_slot,
            )
            await asyncio.wait_for(release_event.wait(), timeout=1)
            monitor = engine.monitors[monitor_id]
            await asyncio.gather(
                monitor._task,
                monitor._expiry_task,
                return_exceptions=True,
            )

        self.assertFalse(monitor.active)
        self.assertEqual(released, [monitor_id])
        self.assertEqual(engine.get_active_monitors(), [])

    def test_proxy_header_is_used_only_when_explicitly_trusted(self):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.1")],
                "client": ("127.0.0.1", 5000),
                "server": ("test", 80),
                "scheme": "http",
                "query_string": b"",
            }
        )

        self.assertEqual(get_client_ip(request, trust_proxy_headers=False), "127.0.0.1")
        self.assertEqual(get_client_ip(request, trust_proxy_headers=True), "203.0.113.7")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI
from starlette.requests import Request

from app.core.rate_limiter import (
    IPRateLimitMiddleware,
    RateLimitRule,
    RateLimitWindow,
    SQLiteIPRateLimiter,
    get_client_ip,
)


class RateLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "limits.db")
        self.limiter = SQLiteIPRateLimiter(self.db_path)
        await self.limiter.initialize()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

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

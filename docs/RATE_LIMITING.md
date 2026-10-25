# Per-IP Rate Limiting

NovaOS protects paid provider credits with a persistent per-IP sliding-window limiter. It is enabled by default and runs before credit-consuming request handlers.

## Protected endpoints

| Endpoint | Short window | Daily window | Purpose |
|---|---:|---:|---|
| `POST /api/command` | 10/minute | 100/day | Protects the multi-agent intelligence pipeline and its provider calls |
| `POST /api/monitor` | 2/hour | 4/day | Prevents repeated creation of continuous background workloads |
| `POST /api/auth/send-otp` | 5/10 minutes | 20/day | Prevents email/OTP abuse |

Only one continuous monitor may be active per IP by default. Monitor intervals are restricted to 15–1,440 minutes and duration to 1–24 hours, preventing zero-delay or unbounded jobs.

All limits are configurable:

```env
NOVA_RATE_LIMIT_ENABLED=true
NOVA_RATE_LIMIT_PEPPER=generate-a-private-random-value-of-at-least-32-characters
NOVA_COMMANDS_PER_MINUTE=10
NOVA_COMMANDS_PER_DAY=100
NOVA_MONITORS_PER_HOUR=2
NOVA_MONITORS_PER_DAY=4
NOVA_MAX_ACTIVE_MONITORS_PER_IP=1
NOVA_OTPS_PER_TEN_MINUTES=5
NOVA_OTPS_PER_DAY=20
```

`NOVA_RATE_LIMIT_PEPPER` is required when the limiter is enabled. Generate it with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and keep it private. NovaOS refuses to start with a missing or shorter value rather than producing predictable IP hashes.

## Client behavior

For example, an allowed `POST /api/command` request returns command-quota headers like:

```text
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1785840000
X-RateLimit-Policy: 10;w=60, 100;w=86400
```

An over-quota `POST /api/command` request returns `429 Too Many Requests` with the same command-quota headers plus `Retry-After`:

```json
{
  "detail": "This IP address has reached its fair-use limit. Try again in 42 seconds.",
  "code": "rate_limit_exceeded",
  "retry_after": 42
}
```

## Storage and privacy

Rate-limit events and active-monitor leases are stored in `nova_rate_limits.db` within NovaOS's writable runtime database directory. SQLite transactions make both quota increments and monitor-slot reservations atomic across concurrent requests and workers that share the database. Raw IP addresses are never stored; the limiter saves an HMAC-SHA-256 digest keyed with the private `NOVA_RATE_LIMIT_PEPPER`.

Set `NOVA_DB_DIR` when the default application directory is not writable.

## Reverse proxies and deployments

By default NovaOS uses the ASGI connection address and ignores spoofable forwarding headers. If the app runs exclusively behind a trusted reverse proxy or CDN that replaces incoming forwarding headers, enable:

```env
NOVA_TRUST_PROXY_HEADERS=true
```

NovaOS then resolves the first valid address from `X-Forwarded-For`, followed by `X-Real-IP` or `X-Vercel-Forwarded-For`. Vercel deployments enable this behavior automatically unless the variable is explicitly set.

Active-monitor slots use expiring database leases, so workers sharing the SQLite file enforce one combined per-IP limit and crashed workers cannot reserve a slot forever. For a horizontally scaled deployment where instances do not share a filesystem, place the database on shared storage, enforce the same limits at the API gateway/CDN, or replace the storage adapter with a shared transactional store. Do not rely on forwarding headers when the application can also be reached directly.

## Disabling limits locally

For controlled development only:

```env
NOVA_RATE_LIMIT_ENABLED=false
```

Keep the limiter enabled in production because disabling it allows a single client to consume the shared provider-credit pool.

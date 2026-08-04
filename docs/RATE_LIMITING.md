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
NOVA_COMMANDS_PER_MINUTE=10
NOVA_COMMANDS_PER_DAY=100
NOVA_MONITORS_PER_HOUR=2
NOVA_MONITORS_PER_DAY=4
NOVA_MAX_ACTIVE_MONITORS_PER_IP=1
NOVA_OTPS_PER_TEN_MINUTES=5
NOVA_OTPS_PER_DAY=20
NOVA_RATE_LIMIT_SALT=replace_with_a_long_random_value
```

## Client behavior

Allowed requests return:

```text
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1785840000
X-RateLimit-Policy: 10;w=60, 100;w=86400
```

Requests over quota return `429 Too Many Requests` with the same headers plus `Retry-After`:

```json
{
  "detail": "This IP address has reached its fair-use limit. Try again in 42 seconds.",
  "code": "rate_limit_exceeded",
  "retry_after": 42
}
```

## Storage and privacy

Rate-limit events are stored in `nova_rate_limits.db` within NovaOS's writable runtime database directory. SQLite transactions make the check-and-increment operation atomic across concurrent requests and local workers. Raw IP addresses are never stored; the limiter saves a SHA-256 hash salted with `NOVA_RATE_LIMIT_SALT`.

Set `NOVA_DB_DIR` when the default application directory is not writable.

## Reverse proxies and deployments

By default NovaOS uses the ASGI connection address and ignores spoofable forwarding headers. If the app runs exclusively behind a trusted reverse proxy or CDN that replaces incoming forwarding headers, enable:

```env
NOVA_TRUST_PROXY_HEADERS=true
```

NovaOS then resolves the first valid address from `X-Forwarded-For`, followed by `X-Real-IP` or `X-Vercel-Forwarded-For`. Vercel deployments enable this behavior automatically unless the variable is explicitly set.

The SQLite store is shared across workers on one filesystem. For a horizontally scaled deployment where instances do not share a filesystem, enforce the same limits at the API gateway/CDN or replace the storage adapter with a shared transactional store. Do not rely on forwarding headers when the application can also be reached directly.

## Disabling limits locally

For controlled development only:

```env
NOVA_RATE_LIMIT_ENABLED=false
```

Keep the limiter enabled in production because disabling it allows a single client to consume the shared provider-credit pool.

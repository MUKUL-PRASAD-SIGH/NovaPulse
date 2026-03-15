import urllib.request, urllib.error, json

def post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

BASE = "http://localhost:8000/api"

print("=" * 50)
print("NOVA FULL SYSTEM CHECK")
print("=" * 50)

# 1. Health
s, r = get(f"{BASE}/health")
print(f"[{'OK' if s==200 else 'FAIL'}] Health: {r}")

# 2. Capabilities
s, r = get(f"{BASE}/capabilities")
print(f"[{'OK' if s==200 else 'FAIL'}] Capabilities: engine={r.get('engine')}")

# 3. Command - news only
print("\n--- Testing /command ---")
s, r = post(f"{BASE}/command", {"text": "tesla", "feature_toggles": {"news": True}})
news_count = len(r.get("result", {}).get("data", {}).get("news", []))
print(f"[{'OK' if s==200 and news_count>0 else 'FAIL'}] News only: {news_count} articles, status={s}")

# 4. Command - all core toggles
s, r = post(f"{BASE}/command", {
    "text": "tesla",
    "feature_toggles": {"news": True, "summary": True, "sentiment": True, "trends": True}
})
data = r.get("result", {}).get("data", {})
tools = [t["tool"] for t in r.get("result", {}).get("tools_executed", [])]
print(f"[{'OK' if s==200 else 'FAIL'}] Core toggles: status={s}")
print(f"  Tools ran: {tools}")
print(f"  Data keys: {list(data.keys())}")
for key in ["news", "summary", "sentiment", "trends"]:
    present = key in data and data[key]
    print(f"  {'OK' if present else 'MISSING'}: {key}")

# 5. Command - MAS toggles
s, r = post(f"{BASE}/command", {
    "text": "tesla",
    "feature_toggles": {"news": True, "entities": True, "social": True, "research": True}
})
data = r.get("result", {}).get("data", {})
tools = [t["tool"] for t in r.get("result", {}).get("tools_executed", [])]
print(f"\n[{'OK' if s==200 else 'FAIL'}] MAS toggles: status={s}")
print(f"  Tools ran: {tools}")
for key in ["entities", "social", "research"]:
    present = key in data and data[key]
    print(f"  {'OK' if present else 'MISSING'}: {key}")

# 6. Export
s, r = post(f"{BASE}/export", {"data": {"news": [{"title": "t", "link": "http://x.com", "source": "x"}]}, "format": "json", "filename": "test"})
print(f"\n[{'OK' if s==200 else 'FAIL'}] Export: status={s}")

# 7. Errors summary
errors = r.get("result", {}).get("errors", []) if isinstance(r, dict) else []
print(f"\n[{'OK' if not errors else 'WARN'}] Errors: {errors or 'none'}")

print("\n" + "=" * 50)
print("CHECK COMPLETE")
print("=" * 50)

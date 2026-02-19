"""Diagnostic: test each import and write results to file."""
import sys
import traceback

results = []

def check(label, fn):
    try:
        fn()
        results.append(f"OK  : {label}")
    except Exception as e:
        results.append(f"FAIL: {label} -> {e}\n{traceback.format_exc()}")

check("fastapi", lambda: __import__("fastapi"))
check("uvicorn", lambda: __import__("uvicorn"))
check("boto3", lambda: __import__("boto3"))
check("httpx", lambda: __import__("httpx"))
check("dotenv", lambda: __import__("dotenv"))
check("pydantic", lambda: __import__("pydantic"))

check("app", lambda: __import__("app"))
check("app.models.schemas", lambda: __import__("app.models.schemas"))
check("app.memory.store", lambda: __import__("app.memory.store"))
check("app.core.tool_registry", lambda: __import__("app.core.tool_registry"))
check("app.core.plan_validator", lambda: __import__("app.core.plan_validator"))
check("app.tools.exporter", lambda: __import__("app.tools.exporter"))
check("app.tools.multi_fetcher", lambda: __import__("app.tools.multi_fetcher"))
check("app.tools.summarizer", lambda: __import__("app.tools.summarizer"))
check("app.tools.sentiment", lambda: __import__("app.tools.sentiment"))
check("app.tools.trends", lambda: __import__("app.tools.trends"))
check("app.agents.planner_agent", lambda: __import__("app.agents.planner_agent"))
check("app.agents.executor_agent", lambda: __import__("app.agents.executor_agent"))
check("app.api.routes", lambda: __import__("app.api.routes"))
check("app.main", lambda: __import__("app.main"))

with open("diagnose_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("\n".join(results))

"""Nova v3 — News Pipeline Node.

Wraps the existing executor_agent.py logic to run the core
news intelligence pipeline (news → summary → sentiment → trends → scraper → entities → images).

This is the PRIMARY pipeline — it runs for every query.
The existing execute_plan() function is reused as-is, so zero logic is rewritten.
"""

import time
from typing import Dict, Any

from app.agents.executor_agent import execute_plan
from app.models.schemas import TaskPlan
from app.memory.store import log


async def news_pipeline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the core news intelligence pipeline.
    
    Reads:  plan, query
    Writes: news_result, execution_trace
    
    Delegates entirely to the v2 executor_agent.execute_plan(),
    which handles:
    - Dependency-aware execution
    - Retry with exponential backoff
    - Tool fallbacks
    - Auto step regeneration
    - Context passing between tools
    """
    start = time.time()

    plan_dict = state.get("plan", {})
    query = state.get("query", "")

    try:
        # Convert plan dict to TaskPlan pydantic model
        task_plan = TaskPlan(
            intent=plan_dict.get("intent", f"Analyze: {query}"),
            domain=plan_dict.get("domain", "general"),
            steps=plan_dict.get("steps", [])
        )

        # Delegate to existing executor — it handles everything internally
        result = execute_plan(task_plan)

        duration = round((time.time() - start) * 1000)

        trace_entry = {
            "node": "news_pipeline",
            "status": "success" if result.get("success") else "partial",
            "duration_ms": duration,
            "tools_executed": len(result.get("tools_executed", [])),
            "errors": len(result.get("errors", [])),
            "skipped": len(result.get("skipped", [])),
            "fallbacks": len(result.get("fallbacks_used", [])),
        }

        log("INFO", f"News pipeline complete in {duration}ms", trace_entry)

        return {
            "news_result": result,
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

    except Exception as e:
        duration = round((time.time() - start) * 1000)
        error_msg = f"News pipeline crashed: {str(e)}"
        log("ERROR", error_msg)

        trace_entry = {
            "node": "news_pipeline",
            "status": "error",
            "duration_ms": duration,
            "error": error_msg,
        }

        return {
            "news_result": {
                "intent": plan_dict.get("intent", "unknown"),
                "domain": plan_dict.get("domain", "unknown"),
                "tools_executed": [],
                "data": {},
                "errors": [error_msg],
                "success": False,
            },
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

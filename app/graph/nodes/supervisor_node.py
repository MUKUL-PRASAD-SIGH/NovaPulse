"""Nova v3 — Supervisor Node.

The meta-controller that decides:
1. Analysis depth (quick / standard / deep)
2. Which pipelines to activate
3. Whether to enable the critic quality gate

This is the ENTRY POINT of the graph.
"""

import re
import time
from typing import Dict, Any

from app.agents.planner_agent import plan_task
from app.memory.store import log


# Keywords that signal depth levels
DEEP_KEYWORDS = {
    "deep", "detailed", "comprehensive", "full", "analyze", "analysis",
    "investigate", "thorough", "complete", "everything", "all"
}
QUICK_KEYWORDS = {
    "quick", "fast", "brief", "just", "only", "simple", "latest"
}


def classify_depth(query: str, toggles: Dict[str, bool]) -> str:
    """Classify the analysis depth based on query and toggle count."""
    query_lower = query.lower()
    words = set(re.findall(r'\b\w+\b', query_lower))

    # Explicit signals
    if words & DEEP_KEYWORDS:
        return "deep"
    if words & QUICK_KEYWORDS:
        return "quick"

    # Toggle count heuristic (only if toggles were provided)
    if toggles:
        enabled_count = sum(1 for v in toggles.values() if v)
        if enabled_count >= 6:
            return "deep"
        elif enabled_count <= 2:
            return "quick"

    return "standard"


def select_pipelines(query: str, toggles: Dict[str, bool], depth: str) -> list:
    """Determine which pipelines to activate."""
    pipelines = ["news"]  # News always runs

    query_lower = query.lower()

    # Social pipeline
    if toggles.get("social", False) or toggles.get("social_monitor", False) or \
       any(w in query_lower for w in ["social", "reddit", "twitter", "buzz"]):
        pipelines.append("social")

    # Research pipeline
    if toggles.get("research", False) or toggles.get("research_assistant", False) or \
       any(w in query_lower for w in ["research", "paper", "academic", "github"]):
        pipelines.append("research")

    # Market pipeline (activated on deep or financial queries)
    if depth == "deep" or \
       any(w in query_lower for w in ["market", "stock", "finance", "trade", "economy"]):
        pipelines.append("market")

    return pipelines


async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Meta-agent: classifies query, sets strategy, generates plan.
    
    Reads:  query, feature_toggles
    Writes: depth, active_pipelines, enable_critic, max_retries, plan, execution_trace
    """
    start = time.time()

    query = state.get("query", "")
    toggles = state.get("feature_toggles", {})

    # 1. Classify depth
    depth = classify_depth(query, toggles)

    # 2. Select pipelines
    pipelines = select_pipelines(query, toggles, depth)

    # 3. Generate plan via existing planner
    plan = plan_task(query)

    # 4. Set strategy
    duration = round((time.time() - start) * 1000)
    trace_entry = {
        "node": "supervisor",
        "status": "success",
        "duration_ms": duration,
        "depth": depth,
        "pipelines": pipelines,
    }

    log("INFO", f"Supervisor: depth={depth}, pipelines={pipelines}")

    return {
        "depth": depth,
        "active_pipelines": pipelines,
        "enable_critic": depth == "deep",
        "max_retries": 2 if depth == "deep" else 0,
        "plan": plan,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
    }

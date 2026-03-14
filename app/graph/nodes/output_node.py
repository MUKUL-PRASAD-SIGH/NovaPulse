"""Nova v3 — Output Node.

Final node — packages the fused report into the API response format.
Adds execution metadata, confidence scoring, and quality badges.
"""

import time
from typing import Dict, Any, List

from app.memory.store import log


def _calculate_confidence(state: Dict[str, Any]) -> float:
    """Calculate overall confidence score (0.0 - 1.0)."""
    factors = []

    # Factor 1: Critic score (biggest weight)
    critic_score = state.get("critic_score", 50)
    factors.append(critic_score / 100.0 * 0.4)

    # Factor 2: Pipeline count
    report = state.get("fused_report", {})
    pipeline_count = report.get("pipeline_count", 1)
    factors.append(min(1.0, pipeline_count / 4.0) * 0.2)

    # Factor 3: Section count
    sections = sum(1 for k in ["news", "summary", "sentiment", "trends",
                                "entities", "social", "research", "market"]
                   if report.get(k))
    factors.append(min(1.0, sections / 6.0) * 0.2)

    # Factor 4: No errors in news pipeline
    news = state.get("news_result", {})
    errors = news.get("errors", []) if isinstance(news, dict) else []
    error_penalty = max(0, 0.2 - len(errors) * 0.05)
    factors.append(error_penalty)

    return round(sum(factors), 2)


def _get_quality_badge(confidence: float, pipeline_count: int) -> str:
    """Determine quality badge for the report."""
    if confidence >= 0.75 and pipeline_count >= 3:
        return "full"       # 🟢 Full Intelligence
    elif confidence >= 0.5 and pipeline_count >= 2:
        return "standard"   # 🟡 Standard Intelligence
    elif confidence >= 0.3:
        return "partial"    # 🟠 Partial Intelligence
    else:
        return "raw"        # 🔴 Raw Data Only


async def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Package the final intelligence report.
    
    Reads:  fused_report, execution_trace, critic_score, plan, query
    Writes: final_report, confidence_score
    """
    start = time.time()

    report = state.get("fused_report", {})
    plan = state.get("plan", {})
    trace = state.get("execution_trace", [])
    depth = state.get("depth", "standard")

    # Calculate confidence
    confidence = _calculate_confidence(state)

    # Quality badge
    pipeline_count = report.get("pipeline_count", 1)
    quality_badge = _get_quality_badge(confidence, pipeline_count)

    # Build the final API-compatible response
    # Maintains backward compatibility with v2 response format
    final = {
        # === v2 COMPATIBLE FIELDS (frontend expects these) ===
        "intent": plan.get("intent", ""),
        "domain": plan.get("domain", ""),
        "tools_executed": report.get("news_meta", {}).get("tools_executed", []),
        "data": {
            "news": report.get("news", []),
            "summary": report.get("summary"),
            "sentiment": report.get("sentiment"),
            "trends": report.get("trends"),
            "entities": report.get("entities"),
            "images": report.get("images"),
            "social": report.get("social"),
            "research": report.get("research"),
            "scraped_articles": report.get("scraped_articles"),
            "market": report.get("market"),
        },
        "errors": report.get("news_meta", {}).get("errors", []),
        "skipped": report.get("news_meta", {}).get("skipped", []),
        "fallbacks_used": report.get("news_meta", {}).get("fallbacks_used", []),
        "success": True,

        # === v3 NEW FIELDS ===
        "v3_meta": {
            "depth": depth,
            "pipelines_executed": report.get("pipelines_included", []),
            "pipeline_count": pipeline_count,
            "critic_score": state.get("critic_score", 100),
            "critic_approved": state.get("critic_approved", True),
            "critic_feedback": state.get("critic_feedback", []),
            "retry_count": state.get("retry_count", 0),
            "confidence": confidence,
            "quality_badge": quality_badge,
            "graph_trace": trace,
            # Phase 3: Memory intelligence
            "memory": {
                "context": state.get("memory_context"),
                "comparison": state.get("memory_comparison"),
            },
        },
    }

    # Remove None values from data
    final["data"] = {k: v for k, v in final["data"].items() if v is not None}

    duration = round((time.time() - start) * 1000)

    trace_entry = {
        "node": "output",
        "status": "success",
        "duration_ms": duration,
    }

    log("INFO", f"Output packaged: confidence={confidence}, badge={quality_badge}")

    return {
        "final_report": final,
        "confidence_score": confidence,
        "execution_trace": trace + [trace_entry],
    }

"""Nova v3 — Memory Node.

Saves the approved intelligence report to ALL memory layers
via the unified MemoryManager. Also records entities and trends
for temporal intelligence.

Phase 3: Full 3-layer memory (short-term + SQLite + semantic stub).
"""

import time
import json
from typing import Dict, Any

from app.memory.manager import memory_manager
from app.memory.store import log


async def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the approved report to memory.
    
    Reads:  fused_report, query, critic_score, confidence_score, 
            active_pipelines, depth, execution_trace
    Writes: memory_context, execution_trace
    """
    start = time.time()

    report = state.get("fused_report", {})
    query = state.get("query", "")
    score = state.get("critic_score", 0)
    confidence = state.get("confidence_score", 0.0)
    pipelines = state.get("active_pipelines", [])
    depth = state.get("depth", "standard")

    try:
        # Extract entities from result for timeline tracking
        entities = _extract_entities(report)

        # Extract trends for snapshot
        trends = _extract_trends(report)

        # Calculate duration so far
        trace = state.get("execution_trace", [])
        total_ms = sum(t.get("duration_ms", 0) for t in trace)

        # Store across all memory layers
        memory_manager.store(
            query=query,
            result=report,
            depth=depth,
            pipelines=pipelines,
            critic_score=score,
            confidence=confidence,
            entities=entities,
            duration_ms=total_ms,
        )

        # Record trends if present
        if trends:
            memory_manager.record_trends(trends)

        # Retrieve comparison context for output
        comparison = memory_manager.compare_topic(query, days=7)

        duration = round((time.time() - start) * 1000)

        trace_entry = {
            "node": "memory",
            "status": "success",
            "duration_ms": duration,
            "entities_stored": len(entities),
            "trends_stored": len(trends),
        }

        log("INFO", f"Memory saved: query='{query}', score={score}, "
            f"entities={len(entities)}, trends={len(trends)}")

        return {
            "memory_context": {
                "stored": True,
                "comparison": comparison,
                "entities_tracked": len(entities),
            },
            "memory_comparison": comparison if comparison.get("comparison_available") else None,
            "execution_trace": trace + [trace_entry],
        }

    except Exception as e:
        duration = round((time.time() - start) * 1000)
        error_msg = f"Memory save error: {str(e)}"
        log("WARN", error_msg)

        return {
            "memory_context": {"stored": False, "error": error_msg},
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "memory",
                "status": "error",
                "duration_ms": duration,
                "error": error_msg,
            }],
        }


def _extract_entities(report: Dict) -> list:
    """Pull entities from the fused report for timeline tracking."""
    entities = []

    # From entity_graph (fusion output)
    entity_graph = report.get("entity_graph", {})
    if isinstance(entity_graph, dict):
        for category, items in entity_graph.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        entities.append(item)
                    elif isinstance(item, str):
                        entities.append({"name": item, "type": category})

    # From pipeline results
    for key in ["news_result", "social_result", "research_result", "market_result"]:
        pipeline = report.get(key, {})
        if isinstance(pipeline, dict):
            pipe_entities = pipeline.get("entities", [])
            if isinstance(pipe_entities, list):
                for ent in pipe_entities:
                    if isinstance(ent, dict) and ent.get("name"):
                        entities.append(ent)

    return entities[:50]  # Cap


def _extract_trends(report: Dict) -> list:
    """Pull trending topics from the report for snapshots."""
    trends = []

    # From direct trends data
    trend_data = report.get("trends", {})
    if isinstance(trend_data, dict):
        for key in ["trending_topics", "rising_topics"]:
            items = trend_data.get(key, [])
            if isinstance(items, list):
                trends.extend(items)

    # From pipeline results
    for key in ["news_result", "market_result"]:
        pipeline = report.get(key, {})
        if isinstance(pipeline, dict):
            pipe_trends = pipeline.get("trends", {})
            if isinstance(pipe_trends, dict):
                for tkey in ["trending_topics", "rising_topics"]:
                    items = pipe_trends.get(tkey, [])
                    if isinstance(items, list):
                        trends.extend(items)

    return trends[:30]  # Cap

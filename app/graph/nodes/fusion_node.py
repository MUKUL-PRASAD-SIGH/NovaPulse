"""Nova v3 — Intelligence Fusion Node.

Combines results from all pipelines into a single unified report.

Phase 1: Simple merge (combine pipeline outputs into one dict)
Phase 2+: LLM-powered cross-pipeline synthesis with contradiction detection
"""

import time
from typing import Dict, Any, Optional

from app.memory.store import log


def _merge_entities(news_entities: Optional[Dict], social_entities: Optional[Dict]) -> Optional[Dict]:
    """Merge entities from multiple pipelines (deduplicated)."""
    if not news_entities and not social_entities:
        return None
    if not social_entities:
        return news_entities
    if not news_entities:
        return social_entities

    # Simple merge — combine entity lists
    merged = dict(news_entities) if isinstance(news_entities, dict) else {}

    if isinstance(social_entities, dict):
        social_ents = social_entities.get("entities", {})
        merged_ents = merged.get("entities", {})

        for category in ["people", "organizations", "locations"]:
            existing = {e.get("name", "").lower() for e in merged_ents.get(category, [])}
            for entity in social_ents.get(category, []):
                if entity.get("name", "").lower() not in existing:
                    merged_ents.setdefault(category, []).append(entity)

        merged["entities"] = merged_ents
        merged["cross_pipeline"] = True

    return merged


async def fusion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Combine all pipeline results into a unified intelligence report.
    
    Reads:  news_result, social_result, research_result, market_result
    Writes: fused_report, execution_trace
    """
    start = time.time()

    news = state.get("news_result")
    social = state.get("social_result")
    research = state.get("research_result")
    market = state.get("market_result")

    # Critic feedback from previous retry (if any)
    feedback = state.get("critic_feedback", [])

    # Build fused report
    fused = {}
    pipelines_included = []

    # === NEWS (primary — from executor result) ===
    if news and isinstance(news, dict):
        data = news.get("data", {})
        fused["news"] = data.get("news", [])
        fused["summary"] = data.get("summary")
        fused["sentiment"] = data.get("sentiment")
        fused["trends"] = data.get("trends")
        fused["scraped_articles"] = data.get("scraped_articles")
        fused["images"] = data.get("images")

        # Entities from news pipeline
        news_entities = data.get("entities")
        pipelines_included.append("news")

        # Execution metadata from news pipeline
        fused["news_meta"] = {
            "tools_executed": news.get("tools_executed", []),
            "errors": news.get("errors", []),
            "skipped": news.get("skipped", []),
            "fallbacks_used": news.get("fallbacks_used", []),
        }
    else:
        news_entities = None

    # === SOCIAL ===
    if social and isinstance(social, dict) and social.get("success"):
        fused["social"] = social.get("social", {})
        social_entities = social.get("entities")
        pipelines_included.append("social")
    else:
        social_entities = None

    # === RESEARCH ===
    if research and isinstance(research, dict) and research.get("success"):
        fused["research"] = research.get("research", {})
        pipelines_included.append("research")

    # === MARKET ===
    if market and isinstance(market, dict) and market.get("success"):
        fused["market"] = {
            "news": market.get("news", []),
            "trends": market.get("trends"),
            "sentiment": market.get("sentiment"),
        }
        pipelines_included.append("market")

    # === MERGE ENTITIES ACROSS PIPELINES ===
    fused["entities"] = _merge_entities(news_entities, social_entities)

    # === METADATA ===
    fused["pipelines_included"] = pipelines_included
    fused["pipeline_count"] = len(pipelines_included)

    # Clean None values
    fused = {k: v for k, v in fused.items() if v is not None}

    duration = round((time.time() - start) * 1000)

    trace_entry = {
        "node": "fusion",
        "status": "success",
        "duration_ms": duration,
        "pipelines_fused": pipelines_included,
    }

    log("INFO", f"Fusion complete: {pipelines_included} in {duration}ms")

    return {
        "fused_report": fused,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
    }

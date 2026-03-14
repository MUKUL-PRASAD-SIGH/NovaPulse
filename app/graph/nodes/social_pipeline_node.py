"""Nova v3 — Social Intelligence Pipeline Node.

Runs social media monitoring independently of the news pipeline.
Pipeline: social_monitor → sentiment analysis on social data
"""

import time
import asyncio
from typing import Dict, Any

from app.tools.social_monitor import monitor_social_media
from app.tools.entity_extractor import extract_entities
from app.memory.store import log


def _clean_topic(query: str) -> str:
    import re
    query = re.sub(r"[^\w\s]", "", query)
    stopwords = {"the", "latest", "news", "give", "me", "show", "tell", "about", "for", "on", "in", "testing", "test"}
    words = [w for w in query.split() if w.lower() not in stopwords]
    clean = " ".join(words).strip()
    return clean if clean else "Technology"

async def social_pipeline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the social intelligence pipeline.
    
    Reads:  query, active_pipelines
    Writes: social_result, execution_trace
    """
    # Skip if not activated by supervisor
    if "social" not in state.get("active_pipelines", []):
        return {
            "social_result": None,
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "social_pipeline",
                "status": "skipped",
                "duration_ms": 0,
            }],
        }

    start = time.time()
    query = state.get("query", "AI")

    try:
        topic = _clean_topic(query)
        # 1. Monitor social media (Reddit, Twitter)
        social_data = await monitor_social_media(
            topic=topic,
            platforms=["reddit"]
        )

        # 2. Extract entities from social posts (lightweight)
        posts_text = ""
        if isinstance(social_data, dict):
            reddit = social_data.get("reddit", {})
            posts = reddit.get("posts", [])
            posts_text = " ".join(
                p.get("title", "") for p in posts[:10]
            )

        entities = None
        if posts_text:
            try:
                entities = await extract_entities(posts_text)
            except Exception:
                pass  # Entity extraction is optional here

        duration = round((time.time() - start) * 1000)

        result = {
            "social": social_data,
            "entities": entities,
            "success": True,
        }

        trace_entry = {
            "node": "social_pipeline",
            "status": "success",
            "duration_ms": duration,
        }

        log("INFO", f"Social pipeline complete in {duration}ms")

        return {
            "social_result": result,
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

    except Exception as e:
        duration = round((time.time() - start) * 1000)
        error_msg = f"Social pipeline error: {str(e)}"
        log("ERROR", error_msg)

        return {
            "social_result": {"social": {}, "success": False, "error": error_msg},
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "social_pipeline",
                "status": "error",
                "duration_ms": duration,
                "error": error_msg,
            }],
        }

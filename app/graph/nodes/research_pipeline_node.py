"""Nova v3 — Research Intelligence Pipeline Node.

Runs academic/technical research independently of news.
Pipeline: research_assistant → summarizer (research-specific)
"""

import time
from typing import Dict, Any

from app.tools.research_assistant import comprehensive_research
from app.memory.store import log


def _clean_topic(query: str) -> str:
    import re
    query = re.sub(r"[^\w\s]", "", query)
    stopwords = {"the", "latest", "news", "give", "me", "show", "tell", "about", "for", "on", "in", "testing", "test"}
    words = [w for w in query.split() if w.lower() not in stopwords]
    clean = " ".join(words).strip()
    return clean if clean else "Artificial Intelligence"

async def research_pipeline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the research intelligence pipeline.
    
    Reads:  query, active_pipelines
    Writes: research_result, execution_trace
    """
    if "research" not in state.get("active_pipelines", []):
        return {
            "research_result": None,
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "research_pipeline",
                "status": "skipped",
                "duration_ms": 0,
            }],
        }

    start = time.time()
    query = state.get("query", "AI")

    try:
        topic = _clean_topic(query)
        # Run comprehensive research (arXiv + GitHub + StackOverflow)
        research_data = await comprehensive_research(topic=topic)

        duration = round((time.time() - start) * 1000)

        result = {
            "research": research_data,
            "success": True,
        }

        trace_entry = {
            "node": "research_pipeline",
            "status": "success",
            "duration_ms": duration,
        }

        log("INFO", f"Research pipeline complete in {duration}ms")

        return {
            "research_result": result,
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

    except Exception as e:
        duration = round((time.time() - start) * 1000)
        error_msg = f"Research pipeline error: {str(e)}"
        log("ERROR", error_msg)

        return {
            "research_result": {"research": {}, "success": False, "error": error_msg},
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "research_pipeline",
                "status": "error",
                "duration_ms": duration,
                "error": error_msg,
            }],
        }

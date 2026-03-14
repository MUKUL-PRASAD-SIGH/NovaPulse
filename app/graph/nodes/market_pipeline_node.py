"""Nova v3 — Market Intelligence Pipeline Node.

Focused on financial/market analysis with trend velocity and
sentiment specifically tuned for market data.
Pipeline: news_fetcher (financial) → trends → sentiment → memory_compare
"""

import time
from typing import Dict, Any

from app.tools.multi_fetcher import fetch_news_multi
from app.tools.trends import extract_trends
from app.tools.sentiment import analyze_sentiment
from app.memory.store import log


async def market_pipeline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the market intelligence pipeline.
    
    Reads:  query, active_pipelines
    Writes: market_result, execution_trace
    """
    if "market" not in state.get("active_pipelines", []):
        return {
            "market_result": None,
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "market_pipeline",
                "status": "skipped",
                "duration_ms": 0,
            }],
        }

    start = time.time()
    query = state.get("query", "market")

    # Enhance query with financial context
    market_query = f"{query} market stock finance"

    try:
        # 1. Fetch financial news (sync function)
        articles = fetch_news_multi(topic=market_query, limit=5)
        if not isinstance(articles, list):
            articles = []

        # 2. Extract market trends
        trends = None
        if articles:
            try:
                trends = extract_trends(news_items=articles)
            except Exception as e:
                log("WARN", f"Market trends extraction failed: {e}")

        # 3. Market sentiment (sync function)
        sentiment = None
        if articles:
            try:
                sentiment = analyze_sentiment(news_items=articles)
            except Exception as e:
                log("WARN", f"Market sentiment failed: {e}")

        duration = round((time.time() - start) * 1000)

        result = {
            "news": articles,
            "trends": trends,
            "sentiment": sentiment,
            "success": True,
        }

        trace_entry = {
            "node": "market_pipeline",
            "status": "success",
            "duration_ms": duration,
        }

        log("INFO", f"Market pipeline complete in {duration}ms")

        return {
            "market_result": result,
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

    except Exception as e:
        duration = round((time.time() - start) * 1000)
        error_msg = f"Market pipeline error: {str(e)}"
        log("ERROR", error_msg)

        return {
            "market_result": {"success": False, "error": error_msg},
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "market_pipeline",
                "status": "error",
                "duration_ms": duration,
                "error": error_msg,
            }],
        }

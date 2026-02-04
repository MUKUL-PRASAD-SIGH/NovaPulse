"""Tool registry for Nova Intelligence Agent."""
from typing import Callable, Dict, List, Optional

from app.tools.news_fetcher import fetch_news
from app.tools.summarizer import summarize_news
from app.tools.sentiment import analyze_sentiment
from app.tools.trends import extract_trends
from app.tools.exporter import export_data

TOOLS: Dict[str, Callable] = {
    "news_fetcher": fetch_news,
    "summarizer": summarize_news,
    "sentiment": analyze_sentiment,
    "trends": extract_trends,
    "exporter": export_data,
}

TOOL_DESCRIPTIONS = {
    "news_fetcher": "Fetches news from RSS. Params: topic, sources, limit",
    "summarizer": "AI summary of news. Params: news_items",
    "sentiment": "Sentiment analysis. Params: news_items",
    "trends": "Extract trending topics. Params: news_items",
    "exporter": "Save to file. Params: data, filename, format",
}

def get_tool(name: str) -> Optional[Callable]:
    return TOOLS.get(name)

def list_tools() -> List[str]:
    return list(TOOLS.keys())

def get_tool_descriptions() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())

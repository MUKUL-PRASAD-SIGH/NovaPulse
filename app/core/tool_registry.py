"""Tool registry for Nova Intelligence Agent."""
from typing import Callable, Dict, List, Optional

from app.tools.multi_fetcher import fetch_news_multi
from app.tools.summarizer import summarize_news
from app.tools.sentiment import analyze_sentiment
from app.tools.trends import extract_trends
from app.tools.exporter import export_data
from app.tools.web_scraper import scrape_url, scrape_multiple_urls
from app.tools.entity_extractor import extract_entities, extract_entities_from_articles
from app.tools.image_analyzer import analyze_image, analyze_article_images
from app.tools.social_monitor import monitor_social_media, track_hashtag
from app.tools.research_assistant import search_academic_papers, comprehensive_research

TOOLS: Dict[str, Callable] = {
    "news_fetcher": fetch_news_multi,  # Multi-source news
    "summarizer": summarize_news,
    "sentiment": analyze_sentiment,
    "trends": extract_trends,
    "exporter": export_data,
    "web_scraper": scrape_url,  # NEW: Extract full article content
    "entity_extractor": extract_entities_from_articles,  # NEW: NER & knowledge graphs
    "image_analyzer": analyze_article_images,  # NEW: Image intelligence
    "social_monitor": monitor_social_media,  # NEW: Social media tracking
    "research_assistant": comprehensive_research,  # NEW: Academic & technical research
}

TOOL_DESCRIPTIONS = {
    "news_fetcher": "Fetches news from multiple sources (Tavily, GNews, RSS) in parallel. Params: topic, limit",
    "summarizer": "AI summary of news. Params: news_items",
    "sentiment": "Sentiment analysis. Params: news_items",
    "trends": "Extract trending topics. Params: news_items",
    "exporter": "Save to file. Params: data, filename, format",
    "web_scraper": "Extract full article content from URLs. Params: url or urls",
    "entity_extractor": "Extract entities (people, orgs, locations) and build knowledge graphs. Params: articles",
    "image_analyzer": "Analyze images from articles (metadata, OCR, manipulation detection). Params: articles",
    "social_monitor": "Monitor social media trends (Reddit, Twitter). Params: topic, platforms",
    "research_assistant": "Search academic papers, GitHub repos, StackOverflow. Params: topic",
}

def get_tool(name: str) -> Optional[Callable]:
    return TOOLS.get(name)

def list_tools() -> List[str]:
    return list(TOOLS.keys())

def get_tool_descriptions() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())

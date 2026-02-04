"""RSS fetcher - async version for parallel fetching."""
import asyncio
import feedparser
from typing import List, Dict


RSS_SOURCES = {
    "google": "https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en",
    "techcrunch": "https://techcrunch.com/tag/{topic}/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
}


async def fetch_rss(topic: str, limit: int = 5) -> Dict:
    """
    Fetch news via RSS feeds.
    
    Returns standardized result with source info.
    """
    try:
        # Run feedparser in thread pool (it's blocking)
        loop = asyncio.get_event_loop()
        search_topic = topic.replace(" ", "+")
        url = RSS_SOURCES["google"].format(topic=search_topic)
        
        feed = await loop.run_in_executor(None, feedparser.parse, url)
        
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", "No title"),
                "url": entry.get("link", ""),
                "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                "published": entry.get("published", ""),
                "source_name": "Google News",
                "fetch_source": "rss"
            })
        
        return {
            "source": "rss",
            "success": True,
            "articles": articles,
            "error": None
        }
        
    except Exception as e:
        return {
            "source": "rss",
            "success": False,
            "articles": [],
            "error": str(e)
        }

"""Multi-source news fetcher tool.

Fetches news from multiple RSS sources including:
- Google News
- TechCrunch
- The Verge
"""
import feedparser
from typing import List, Dict


# RSS feed templates - {topic} will be replaced
RSS_SOURCES = {
    "google": "https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en",
    "techcrunch": "https://techcrunch.com/tag/{topic}/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
}

# Topic aliases for better matching
TOPIC_ALIASES = {
    "ai": "artificial+intelligence",
    "ml": "machine+learning",
    "crypto": "cryptocurrency",
    "tech": "technology",
}


def fetch_news(
    topic: str = "ai",
    sources: List[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Fetch news articles from multiple RSS sources.
    
    Args:
        topic: News topic to search for (ai, tech, crypto, etc.)
        sources: List of source names (google, techcrunch, verge)
        limit: Maximum articles per source
    
    Returns:
        List of news items with title, link, source, published
    """
    if sources is None:
        sources = ["google"]
    
    # Normalize topic - replace spaces with + for URL
    search_topic = TOPIC_ALIASES.get(topic.lower(), topic)
    search_topic = search_topic.replace(" ", "+")
    
    all_news = []
    
    for source in sources:
        try:
            url_template = RSS_SOURCES.get(source.lower())
            if not url_template:
                continue
            
            url = url_template.format(topic=search_topic)
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:limit]:
                all_news.append({
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", ""),
                    "source": source,
                    "published": entry.get("published", "")
                })
                
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            continue
    
    return all_news


def get_available_sources() -> List[str]:
    """Return list of available news sources."""
    return list(RSS_SOURCES.keys())

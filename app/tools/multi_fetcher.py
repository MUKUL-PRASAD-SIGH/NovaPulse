"""Multi-source fetcher - simplified sync version."""
import feedparser
import os
import httpx
from typing import List, Dict
from difflib import SequenceMatcher
from dotenv import load_dotenv

from app.memory.store import log

load_dotenv()


def fetch_news_multi(topic: str, limit: int = 5, **kwargs) -> List[Dict]:
    """
    Fetch from multiple sources with failover.
    Simplified sync version for reliability.
    """
    log("INFO", f"Multi-source fetch for: {topic}")
    
    all_articles = []
    sources_status = {}
    
    # 1. Try RSS (always works, free)
    rss_result = _fetch_rss_sync(topic, limit)
    sources_status["rss"] = {"success": rss_result["success"], "count": len(rss_result["articles"])}
    if rss_result["success"]:
        all_articles.extend(rss_result["articles"])
    
    # 2. Try GNews if key exists
    gnews_key = os.getenv("GNEWS_API_KEY", "")
    if gnews_key:
        gnews_result = _fetch_gnews_sync(topic, limit, gnews_key)
        sources_status["gnews"] = {"success": gnews_result["success"], "count": len(gnews_result["articles"])}
        if gnews_result["success"]:
            all_articles.extend(gnews_result["articles"])
    
    # 3. Try Tavily if key exists
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        tavily_result = _fetch_tavily_sync(topic, limit, tavily_key)
        sources_status["tavily"] = {"success": tavily_result["success"], "count": len(tavily_result["articles"])}
        if tavily_result["success"]:
            all_articles.extend(tavily_result["articles"])
    
    # Deduplicate
    unique = _deduplicate(all_articles)
    log("INFO", f"Multi-source complete: {len(unique)} articles from {len(sources_status)} sources")
    
    return unique


def _fetch_rss_sync(topic: str, limit: int) -> Dict:
    """Fetch from Google News RSS."""
    try:
        url = f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", "No title"),
                "link": entry.get("link", ""),
                "source": "rss",
                "published": entry.get("published", "")
            })
        return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"RSS failed: {e}")
        return {"success": False, "articles": []}


def _fetch_gnews_sync(topic: str, limit: int, api_key: str) -> Dict:
    """Fetch from GNews API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://gnews.io/api/v4/search",
                params={"apikey": api_key, "q": topic, "lang": "en", "max": limit}
            )
            if resp.status_code == 429 or resp.status_code == 403:
                return {"success": False, "articles": [], "error": "quota"}
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", [])[:limit]:
                articles.append({
                    "title": item.get("title", "No title"),
                    "link": item.get("url", ""),
                    "source": "gnews",
                    "published": item.get("publishedAt", "")
                })
            return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"GNews failed: {e}")
        return {"success": False, "articles": []}


def _fetch_tavily_sync(topic: str, limit: int, api_key: str) -> Dict:
    """Fetch from Tavily search API with title cleaning."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": f"{topic} breaking news today",
                    "search_depth": "advanced",
                    "max_results": limit + 5,  # Fetch extra to filter
                    "include_domains": ["reuters.com", "bloomberg.com", "cnbc.com", "bbc.com", "cnn.com", "theguardian.com", "apnews.com"]
                }
            )
            if resp.status_code == 429:
                return {"success": False, "articles": [], "error": "quota"}
            resp.raise_for_status()
            data = resp.json()
            articles = []
            
            # Generic titles to skip
            skip_titles = ["latest stories", "today's latest", "news & updates", "latest news", "home", "news hub"]
            
            for r in data.get("results", []):
                if len(articles) >= limit:
                    break
                    
                title = _clean_title(r.get("title", ""))
                
                # Skip generic/useless titles
                if not title or len(title) < 15:
                    continue
                if any(skip in title.lower() for skip in skip_titles):
                    continue
                if title.lower().startswith("by "):  # Byline, not headline
                    continue
                
                articles.append({
                    "title": title,
                    "link": r.get("url", ""),
                    "source": "tavily",
                    "published": ""
                })
            
            return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"Tavily failed: {e}")
        return {"success": False, "articles": []}


def _clean_title(title: str) -> str:
    """Clean web page title to extract main headline."""
    # Remove common separators and site names
    separators = [' | ', ' - ', ' – ', ' — ', ' :: ', ' : ']
    
    for sep in separators:
        if sep in title:
            parts = title.split(sep)
            # Usually the main headline is the first or longest part
            # Filter out parts that look like site names (short or contain common words)
            site_words = ['news', 'reuters', 'bbc', 'cnn', 'wsj', 'times', 'post', 'daily', 'insider']
            main_parts = []
            for p in parts:
                p_lower = p.lower().strip()
                # Skip if it's just a site name
                if len(p_lower) < 15 and any(w in p_lower for w in site_words):
                    continue
                main_parts.append(p.strip())
            
            if main_parts:
                # Return the longest meaningful part
                return max(main_parts, key=len)
    
    return title.strip()


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """Remove duplicates by URL and similar titles."""
    seen_urls = set()
    seen_titles = []
    unique = []
    
    for a in articles:
        url = a.get("link", "")
        title = a.get("title", "")
        
        if url and url in seen_urls:
            continue
        
        is_dup = any(SequenceMatcher(None, title.lower(), t.lower()).ratio() > 0.85 for t in seen_titles)
        if is_dup:
            continue
        
        seen_urls.add(url)
        seen_titles.append(title)
        unique.append(a)
    
    return unique

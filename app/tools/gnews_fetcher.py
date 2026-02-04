"""GNews API fetcher - structured news from publishers."""
import os
import httpx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_URL = "https://gnews.io/api/v4/search"


async def fetch_gnews(topic: str, limit: int = 5) -> Dict:
    """
    Fetch news via GNews API.
    
    Returns standardized result with source info.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    
    if not api_key:
        return {
            "source": "gnews",
            "success": False,
            "articles": [],
            "error": "GNEWS_API_KEY not configured"
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                GNEWS_API_URL,
                params={
                    "apikey": api_key,
                    "q": topic,
                    "lang": "en",
                    "max": limit
                }
            )
            
            if response.status_code == 429:
                return {
                    "source": "gnews",
                    "success": False,
                    "articles": [],
                    "error": "QUOTA_EXCEEDED"
                }
            
            if response.status_code == 403:
                return {
                    "source": "gnews",
                    "success": False,
                    "articles": [],
                    "error": "QUOTA_EXCEEDED"
                }
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("articles", [])[:limit]:
                articles.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "summary": item.get("description", "")[:200],
                    "published": item.get("publishedAt", ""),
                    "source_name": item.get("source", {}).get("name", "Unknown"),
                    "fetch_source": "gnews"
                })
            
            return {
                "source": "gnews",
                "success": True,
                "articles": articles,
                "error": None
            }
            
    except httpx.TimeoutException:
        return {"source": "gnews", "success": False, "articles": [], "error": "TIMEOUT"}
    except Exception as e:
        return {"source": "gnews", "success": False, "articles": [], "error": str(e)}

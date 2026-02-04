"""Tavily web search fetcher - broad coverage search."""
import os
import httpx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_URL = "https://api.tavily.com/search"


async def fetch_tavily(topic: str, limit: int = 5) -> Dict:
    """
    Fetch news via Tavily web search API.
    
    Returns standardized result with source info.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    
    if not api_key:
        return {
            "source": "tavily",
            "success": False,
            "articles": [],
            "error": "TAVILY_API_KEY not configured"
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                TAVILY_API_URL,
                json={
                    "api_key": api_key,
                    "query": f"{topic} news",
                    "search_depth": "basic",
                    "max_results": limit,
                    "include_answer": False
                }
            )
            
            if response.status_code == 429:
                return {
                    "source": "tavily",
                    "success": False,
                    "articles": [],
                    "error": "QUOTA_EXCEEDED"
                }
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for result in data.get("results", [])[:limit]:
                articles.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "summary": result.get("content", "")[:200],
                    "published": "",
                    "source_name": result.get("url", "").split("/")[2] if result.get("url") else "Web",
                    "fetch_source": "tavily"
                })
            
            return {
                "source": "tavily",
                "success": True,
                "articles": articles,
                "error": None
            }
            
    except httpx.TimeoutException:
        return {"source": "tavily", "success": False, "articles": [], "error": "TIMEOUT"}
    except Exception as e:
        return {"source": "tavily", "success": False, "articles": [], "error": str(e)}

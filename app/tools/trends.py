"""Trend extraction tool - extracts trending topics from headlines."""
import re
from collections import Counter
from typing import List, Dict

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "it", "its", "this", "that", "new", "says", "said", "after", "before", "into"}


def extract_trends(news_items: List[Dict]) -> Dict:
    """Extract trending topics from news headlines."""
    if not news_items:
        return {"trending_topics": [], "total_articles": 0}
    
    all_words = []
    for item in news_items:
        title = item.get("title", "")
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        all_words.extend(proper_nouns)
        words = re.findall(r'\b[A-Za-z]{3,}\b', title)
        filtered = [w for w in words if w.lower() not in STOPWORDS]
        all_words.extend(filtered)
    
    top_topics = Counter(all_words).most_common(10)
    
    return {
        "trending_topics": [{"topic": word, "mentions": count} for word, count in top_topics],
        "total_articles": len(news_items)
    }

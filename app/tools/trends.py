"""Trend Intelligence V2 - Time-weighted, phrase-aware, velocity-enabled trend detection."""
import re
import json
import os
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dateutil import parser as date_parser

# Stopwords for filtering - includes generic headline words
STOPWORDS = {
    # Common words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", 
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "have", "has", 
    "had", "do", "does", "did", "will", "would", "could", "should", "it", "its",
    "this", "that", "new", "says", "said", "after", "before", "into", "how", "why",
    "what", "when", "where", "who", "which", "more", "most", "some", "any", "all",
    "just", "now", "live", "update", "updates", "report", "reports", "news",
    # Generic headline words (not meaningful as trends)
    "today", "here", "full", "list", "check", "top", "best", "latest", "breaking",
    "stock", "stocks", "market", "markets", "share", "shares", "price", "prices",
    "gains", "falls", "rises", "drops", "higher", "lower", "points", "percent",
    "highlights", "watch", "focus", "key", "major", "big", "small", "each",
    "among", "over", "under", "about", "lakh", "crore", "investors", "trading",
    "february", "january", "march", "april", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "week", "month", "year", "daily",
    # Generic news/business terms
    "deal", "deals", "trade", "talks", "agreement", "announce", "announced",
    "asia", "pacific", "europe", "africa", "movers", "times", "current", "affairs",
    "magazine", "analyst", "analysts", "experts", "sources", "details", "expected"
}

# Source reliability weights
SOURCE_WEIGHTS = {
    "tavily": 1.5,      # AI-curated search
    "gnews": 1.3,       # Google News API
    "rss": 1.0,         # RSS feeds
    "unknown": 0.8
}

# Trend history file
TRENDS_HISTORY_FILE = "app/memory/trends_history.json"


def extract_trends(news_items: List[Dict]) -> Dict:
    """
    Extract trending topics with V2 intelligence:
    - Time-weighted scoring
    - N-gram phrase detection
    - Trend velocity indicators
    - Source reliability weighting
    """
    if not news_items:
        return {
            "trending_topics": [],
            "rising_topics": [],
            "fading_topics": [],
            "total_articles": 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # Extract weighted topics
    topic_scores = _extract_weighted_topics(news_items)
    
    # Get top topics with scores
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Calculate velocity by comparing with history
    velocity_data = _calculate_velocity(sorted_topics)
    
    # Categorize by velocity
    rising = []
    fading = []
    stable = []
    
    for topic, score in sorted_topics[:10]:
        velocity = velocity_data.get(topic, 0)
        topic_data = {
            "topic": topic,
            "score": round(score, 2),
            "mentions": int(score),  # For backward compatibility
            "velocity": _velocity_label(velocity),
            "velocity_value": velocity
        }
        
        if velocity > 0.3:
            rising.append(topic_data)
        elif velocity < -0.3:
            fading.append(topic_data)
        else:
            stable.append(topic_data)
    
    # Save current snapshot for future velocity calculation
    _save_trends_snapshot(sorted_topics)
    
    # Build final output
    all_topics = []
    for topic, score in sorted_topics[:10]:
        velocity = velocity_data.get(topic, 0)
        all_topics.append({
            "topic": topic,
            "score": round(score, 2),
            "mentions": int(score),
            "velocity": _velocity_label(velocity),
            "velocity_icon": _velocity_icon(velocity)
        })
    
    # Generate LLM-powered insights for top topics
    topic_insights, topic_insights_meta = _generate_llm_topic_insights(
        news_items, [t["topic"] for t in all_topics[:5]]
    )
    
    return {
        "trending_topics": all_topics,
        "rising_topics": rising[:5],
        "fading_topics": fading[:3],
        "total_articles": len(news_items),
        "analysis_timestamp": datetime.now().isoformat(),
        "topic_insights": topic_insights,
        # Debug/status info so the UI (and you) can tell whether Nova actually ran
        "topic_insights_meta": topic_insights_meta,
    }


def _generate_llm_topic_insights(news_items: List[Dict], top_topics: List[str]) -> Tuple[Dict[str, Dict], Dict]:
    """
    Use Nova LLM to analyze headlines and generate contextual insights for each trending topic.
    Returns a dict mapping topic -> {insight, emotion, angle}
    """
    import os
    import json
    import logging
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    
    if not news_items or not top_topics:
        return {}, {"enabled": False, "reason": "no_news_or_topics"}
    
    # NOTE: Trends insights should not be coupled to planner mock mode.
    # Default is enabled so Nova can power the "why trending" insights out of the box.
    if os.getenv("USE_MOCK_TRENDS_INSIGHTS", "false").lower() == "true":
        return {}, {"enabled": False, "reason": "mock_trends_insights"}
    
    try:
        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)
        
        # Build headlines string for context
        headlines = [item.get("title", "") for item in news_items[:15]]
        headlines_text = "\n".join([f"- {h}" for h in headlines if h])
        
        topics_text = ", ".join(top_topics[:5])
        
        prompt = f"""Analyze these news headlines and explain why each topic is trending.

Headlines:
{headlines_text}

Top Topics: {topics_text}

For each topic, provide a brief insight (1 sentence max) explaining WHY it's getting coverage right now.

Return ONLY valid JSON mapping topics to insights:
{{
  "India": {{"insight": "Trade deal signed...", "emotion": "Optimism", "angle": "Positive"}},
  "Trump": {{"insight": "announced tariffs...", "emotion": "Neutral", "angle": "Neutral"}}
}}"""

        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 500, "temperature": 0.5}
        }
        
        response = client.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        
        # Parse JSON from response
        if "{" in output_text and "}" in output_text:
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            json_str = output_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Handle potential wrapper keys
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                # If wrapped like {"topics": {...}} or {"topic_name": {...}}
                first_key = list(data.keys())[0]
                if first_key in ["topics", "topic_name", "insights"]:
                     return data[first_key], {"enabled": True, "reason": "ok", "model": "amazon.nova-lite-v1:0", "region": region}
                # If wrapped like {"India": {...}} but inside another dict (less likely with new prompt)
                
            return data, {"enabled": True, "reason": "ok", "model": "amazon.nova-lite-v1:0", "region": region}
        
        return {}, {"enabled": True, "reason": "no_json_in_response", "model": "amazon.nova-lite-v1:0", "region": region}
    except Exception as e:
        # Fallback to empty if LLM fails (but expose the reason)
        logging.getLogger(__name__).warning("Nova trends insights failed: %s", str(e))
        return {}, {"enabled": True, "reason": "exception", "error": str(e), "model": "amazon.nova-lite-v1:0"}


def fuse_trends_with_sentiment(trends_data: Dict, sentiment_data: Dict) -> Dict:
    """
    Fuse trend intelligence with sentiment analysis to create narrative signals.
    
    Returns enhanced trends with per-topic sentiment correlation and narrative labels.
    """
    if not trends_data or not sentiment_data:
        return trends_data
    
    # Extract sentiment indicators
    market_bias = sentiment_data.get("market_bias", "balanced")
    sentiment_direction = sentiment_data.get("direction", "stable")
    confidence = sentiment_data.get("confidence", "medium")
    positive_signals = sentiment_data.get("positive_signals", [])
    negative_signals = sentiment_data.get("negative_signals", [])
    
    # Convert sentiment to numeric score (-1 to 1)
    sentiment_score = _calculate_sentiment_score(sentiment_data)
    
    # Get source spread from trends
    source_spread = trends_data.get("source_spread", {})
    
    # Enhance trending topics with full narrative intelligence
    enhanced_topics = []
    active_narratives = []
    
    for topic in trends_data.get("trending_topics", []):
        topic_name = topic.get("topic", "")
        velocity = topic.get("velocity", "stable")
        velocity_value = _velocity_to_numeric(velocity)
        score = topic.get("score", 0)
        
        # Calculate fusion score
        fusion_score = (velocity_value + 1) * (sentiment_score + 1) / 2
        
        # Determine news-safe narrative
        narrative = _determine_narrative(velocity_value, sentiment_score, fusion_score)
        
        # Determine news cycle stage
        news_cycle = _determine_news_cycle(velocity_value, score)
        
        # Generate coverage tone description
        tone = _get_tone_description(sentiment_score)
        
        # Generate coverage growth description
        coverage_growth = _get_coverage_growth(velocity)
        
        # Get LLM insights for this topic (from pre-computed data)
        topic_insights = trends_data.get("topic_insights", {})
        llm_insight = topic_insights.get(topic_name, {})
        
        # Generate "Why Trending" - Prioritize Nova LLM insights, fallback to templates only if Nova unavailable
        llm_meta = trends_data.get("topic_insights_meta", {})
        nova_available = llm_meta.get("enabled") and llm_meta.get("reason") == "ok"
        
        if llm_insight.get("insight") and nova_available:
            # Use ONLY Nova LLM insight when available (no generic templates)
            why_trending = [llm_insight["insight"]]
        else:
            # Fallback to template-based reasons only if Nova didn't generate insights
            # (_generate_why_trending will add a note if Nova failed)
            why_trending = _generate_why_trending(
                velocity, score, sentiment_score, source_spread, topic_name, llm_meta
            )
        
        enhanced_topic = {
            **topic,
            # News-friendly labels
            "story_direction": narrative["label"],
            "story_icon": narrative["icon"],
            "coverage_growth": coverage_growth,
            "tone_of_coverage": tone,
            "news_cycle_stage": news_cycle["stage"],
            "news_cycle_icon": news_cycle["icon"],
            "why_trending": why_trending,
            # Keep original for compatibility
            "narrative": narrative["label"],
            "narrative_icon": narrative["icon"],
            "fusion_score": round(fusion_score, 2)
        }
        enhanced_topics.append(enhanced_topic)
        
        # Build active narratives (top stories)
        if score > 5 or velocity_value > 0.3:
            active_narratives.append({
                "topic": topic_name,
                "story_direction": narrative["label"],
                "story_icon": narrative["icon"],
                "coverage": coverage_growth,
                "tone": tone,
                "news_cycle": news_cycle["stage"],
                "why_trending": why_trending
            })
    
    # Generate news narrative summary
    news_summary = _generate_news_narrative_summary(
        enhanced_topics, sentiment_score, trends_data.get("rising_topics", [])
    )
    
    return {
        **trends_data,
        "trending_topics": enhanced_topics,
        "active_narratives": active_narratives[:5],
        "news_narrative_summary": news_summary,
        # Keep for backward compatibility
        "narrative_signals": active_narratives[:5],
        "market_narrative": news_summary
    }


def _calculate_sentiment_score(sentiment: Dict) -> float:
    """Convert sentiment data to numeric score (-1 to 1)."""
    bias = sentiment.get("market_bias", "balanced")
    direction = sentiment.get("direction", "stable")
    
    # Base score from bias
    if bias == "risk_on":
        base = 0.6
    elif bias == "risk_off":
        base = -0.6
    else:
        base = 0
    
    # Adjust by direction
    if direction == "improving":
        base += 0.3
    elif direction == "deteriorating":
        base -= 0.3
    
    return max(-1, min(1, base))


def _velocity_to_numeric(velocity: str) -> float:
    """Convert velocity label to numeric value."""
    mapping = {
        "rising_fast": 1.0,
        "rising": 0.5,
        "stable": 0,
        "fading": -0.5,
        "fading_fast": -1.0
    }
    return mapping.get(velocity, 0)


def _sentiment_alignment(score: float) -> str:
    """Get sentiment alignment label."""
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    return "neutral"


def _determine_narrative(velocity: float, sentiment: float, fusion: float) -> Dict:
    """Determine the narrative signal based on trend and sentiment fusion."""
    if velocity > 0.3 and sentiment > 0.3:
        return {"label": "Positive Momentum", "icon": "🟢"}
    elif velocity > 0.3 and sentiment < -0.3:
        return {"label": "Controversial Coverage", "icon": "🟡"}
    elif velocity < -0.3 and sentiment < -0.3:
        return {"label": "Critical/Risk Coverage", "icon": "🔴"}
    elif velocity < -0.3 and sentiment > 0.3:
        return {"label": "Recovery Watch", "icon": "🟡"}
    elif fusion > 0.6:
        return {"label": "Strong Coverage", "icon": "🟢"}
    elif velocity > 0.5:
        return {"label": "Emerging Story", "icon": "🔵"}
    elif fusion < 0.2:
        return {"label": "Low Coverage", "icon": "⚪"}
    else:
        return {"label": "Stable Coverage", "icon": "⚪"}


def _determine_news_cycle(velocity: float, score: float) -> Dict:
    """Determine news cycle stage based on velocity and score."""
    if velocity > 0.7 and score < 8:
        return {"stage": "Breaking Story", "icon": "🆕"}
    elif velocity > 0.3 and score > 10:
        return {"stage": "Peak Focus", "icon": "🔥"}
    elif velocity < -0.3:
        return {"stage": "Losing Attention", "icon": "📉"}
    elif score > 15:
        return {"stage": "Major Story", "icon": "📰"}
    elif score < 5:
        return {"stage": "Background", "icon": "📚"}
    else:
        return {"stage": "Active Coverage", "icon": "🗞️"}


def _get_tone_description(sentiment_score: float) -> str:
    """Get readable tone description from sentiment score."""
    if sentiment_score > 0.5:
        return "Highly Positive"
    elif sentiment_score > 0.2:
        return "Mostly Positive"
    elif sentiment_score > -0.2:
        return "Neutral/Mixed"
    elif sentiment_score > -0.5:
        return "Mostly Critical"
    else:
        return "Highly Critical"


def _get_coverage_growth(velocity: str) -> str:
    """Get readable coverage growth description."""
    mapping = {
        "rising_fast": "Rising Fast",
        "rising": "Rising",
        "stable": "Steady",
        "fading": "Declining",
        "fading_fast": "Dropping Fast"
    }
    return mapping.get(velocity, "Steady")


def _generate_why_trending(
    velocity: str,
    score: float,
    sentiment: float,
    source_spread: Dict,
    topic: str = "",
    llm_meta: Dict | None = None,
) -> List[str]:
    """Generate dynamic, topic-aware explanations for why topic is trending."""
    import random
    import hashlib
    reasons = []

    # If Nova insights are enabled but failing, surface a clear, non-generic reason first.
    # This prevents "default-y" explanations from masking configuration issues.
    if llm_meta and llm_meta.get("enabled") and llm_meta.get("reason") not in (None, "ok"):
        reasons.append(f"Nova insights unavailable ({llm_meta.get('reason')})")
    
    # Make output stable per topic/run by seeding with topic+velocity+score bucket
    seed_src = f"{topic}|{velocity}|{int(score)}"
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed)

    # Dynamic coverage templates (picked deterministically after seeding)
    rising_templates = [
        f"'{topic}' gaining traction in news cycle",
        "Breaking into mainstream coverage",
        "Multiple outlets picking up this story",
        "Cross-platform media attention detected",
        "Story momentum building rapidly"
    ]
    
    high_score_templates = [
        f"High frequency in recent headlines",
        "Dominant narrative in current news cycle",
        f"'{topic}' mentioned across 10+ articles",
        "Central theme in today's coverage"
    ]
    
    moderate_score_templates = [
        "Steady presence in news feeds",
        "Consistent coverage across sources",
        "Active but not dominant story"
    ]
    
    spike_templates = [
        "Sudden surge in last 6 hours",
        "Breaking story dynamics detected",
        "Rapid acceleration in coverage",
        "Real-time news momentum"
    ]
    
    positive_templates = [
        "Optimistic framing in headlines",
        "Positive sentiment in coverage",
        "Favorable media narrative"
    ]
    
    negative_templates = [
        "Critical tone in reporting",
        "Risk/concern framing detected",
        "Cautionary coverage angle"
    ]
    
    neutral_templates = [
        "Balanced reporting across outlets",
        "Factual coverage without strong stance",
        "Neutral journalistic framing"
    ]
    
    # Build reasons dynamically
    if velocity in ["rising_fast", "rising"]:
        reasons.append(random.choice(rising_templates))
    
    if score > 12:
        reasons.append(random.choice(high_score_templates))
    elif score > 8:
        reasons.append(random.choice(moderate_score_templates))
    
    if velocity == "rising_fast":
        reasons.append(random.choice(spike_templates))
    
    # Sentiment-based reasons
    if sentiment > 0.3:
        reasons.append(random.choice(positive_templates))
    elif sentiment < -0.3:
        reasons.append(random.choice(negative_templates))
    else:
        reasons.append(random.choice(neutral_templates))
    
    # Source spread
    if source_spread:
        source_count = len([v for v in source_spread.values() if v > 0])
        if source_count >= 3:
            reasons.append(f"Covered across {source_count} source types")
    
    # Deduplicate and return
    unique_reasons = list(dict.fromkeys(reasons))
    return unique_reasons[:3] if unique_reasons else ["Active in current news cycle"]


def _generate_news_narrative_summary(topics: List, sentiment_score: float, rising: List) -> str:
    """Generate news-friendly narrative summary."""
    rising_count = len(rising)
    top_topics = [t.get("topic", "") for t in topics[:3]]
    
    if not top_topics:
        return "Limited news activity in this domain"
    
    topic_str = ", ".join(top_topics[:2])
    
    if sentiment_score > 0.4 and rising_count >= 2:
        return f"Active positive coverage on {topic_str} with multiple emerging stories"
    elif sentiment_score > 0.2:
        return f"Positive media attention on {topic_str} themes"
    elif sentiment_score < -0.4:
        return f"Critical coverage environment focusing on {topic_str}"
    elif sentiment_score < -0.2:
        return f"Mixed narrative with cautious coverage on {topic_str}"
    elif rising_count >= 2:
        return f"Active story movement around {topic_str}"
    else:
        return f"Stable coverage with focus on {topic_str}"


def _overall_narrative(sentiment_score: float, rising_topics: List) -> str:
    """Generate overall narrative summary (backward compatibility)."""
    return _generate_news_narrative_summary([], sentiment_score, rising_topics)


def _extract_weighted_topics(news_items: List[Dict]) -> Dict[str, float]:
    """Extract topics with time and source weighting."""
    topic_scores = Counter()
    now = datetime.now()
    
    for item in news_items:
        title = item.get("title", "")
        source = item.get("source", "unknown").lower()
        published = item.get("published", "")
        
        # Calculate time weight
        time_weight = _calculate_time_weight(published, now)
        
        # Get source weight
        source_weight = SOURCE_WEIGHTS.get(source, 0.8)
        
        # Combined weight
        weight = time_weight * source_weight
        
        # Extract single words (proper nouns prioritized)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        for noun in proper_nouns:
            if noun.lower() not in STOPWORDS and len(noun) > 2:
                topic_scores[noun] += weight * 1.5  # Boost proper nouns
        
        # Extract regular words
        words = re.findall(r'\b[A-Za-z]{3,}\b', title)
        for word in words:
            if word.lower() not in STOPWORDS:
                topic_scores[word] += weight
        
        # Extract bigrams (2-word phrases)
        bigrams = _extract_ngrams(title, 2)
        for bigram in bigrams:
            topic_scores[bigram] += weight * 1.3  # Boost phrases
        
        # Extract trigrams (3-word phrases) 
        trigrams = _extract_ngrams(title, 3)
        for trigram in trigrams:
            topic_scores[trigram] += weight * 1.2
    
    # Merge similar topics (basic dedup)
    topic_scores = _merge_similar_topics(topic_scores)
    
    return topic_scores


def _extract_ngrams(text: str, n: int) -> List[str]:
    """Extract n-gram phrases from text."""
    # Clean and tokenize
    words = re.findall(r'\b[A-Za-z]{2,}\b', text)
    words = [w for w in words if w.lower() not in STOPWORDS]
    
    ngrams = []
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i+n])
        # Only include if it starts with uppercase (proper phrase)
        if words[i][0].isupper():
            ngrams.append(phrase)
    
    return ngrams


def _calculate_time_weight(published: str, now: datetime) -> float:
    """Calculate time-based weight. Recent = higher weight."""
    if not published:
        return 1.0  # Default weight if no timestamp
    
    try:
        # Parse the published date
        pub_date = date_parser.parse(published, fuzzy=True)
        
        # Make timezone naive for comparison
        if pub_date.tzinfo:
            pub_date = pub_date.replace(tzinfo=None)
        
        # Calculate hours ago
        delta = now - pub_date
        hours_ago = delta.total_seconds() / 3600
        
        # Time weighting: newer = higher weight
        if hours_ago < 2:
            return 3.0      # Last 2 hours: 3x weight
        elif hours_ago < 6:
            return 2.5      # Last 6 hours: 2.5x weight
        elif hours_ago < 12:
            return 2.0      # Last 12 hours: 2x weight
        elif hours_ago < 24:
            return 1.5      # Last 24 hours: 1.5x weight
        elif hours_ago < 48:
            return 1.0      # Last 2 days: 1x weight
        else:
            return 0.5      # Older: 0.5x weight
            
    except Exception:
        return 1.0  # Default on parse error


def _merge_similar_topics(scores: Counter) -> Counter:
    """Merge similar topics (e.g., 'AI' and 'AI model' into 'AI')."""
    merged = Counter()
    sorted_topics = sorted(scores.items(), key=lambda x: len(x[0]))
    
    merged_into = {}
    
    for topic, score in sorted_topics:
        topic_lower = topic.lower()
        found_parent = False
        
        # Check if this topic is part of a longer phrase we've seen
        for existing in list(merged.keys()):
            if topic_lower in existing.lower() and topic != existing:
                merged_into[topic] = existing
                merged[existing] += score * 0.5  # Partial merge
                found_parent = True
                break
        
        if not found_parent:
            merged[topic] = score
    
    return merged


def _calculate_velocity(current_topics: List[Tuple[str, float]]) -> Dict[str, float]:
    """Calculate trend velocity by comparing with previous snapshot."""
    velocity = {}
    previous = _load_trends_snapshot()
    
    if not previous:
        # No history, all topics are neutral
        return {topic: 0 for topic, _ in current_topics}
    
    prev_scores = {item["topic"]: item["score"] for item in previous}
    
    for topic, current_score in current_topics:
        prev_score = prev_scores.get(topic, 0)
        
        if prev_score == 0:
            # New topic = rising
            velocity[topic] = 1.0
        else:
            # Calculate percentage change
            change = (current_score - prev_score) / prev_score
            velocity[topic] = max(-1.0, min(1.0, change))  # Clamp to [-1, 1]
    
    return velocity


def _velocity_label(velocity: float) -> str:
    """Convert velocity value to label."""
    if velocity > 0.5:
        return "rising_fast"
    elif velocity > 0.2:
        return "rising"
    elif velocity < -0.5:
        return "fading_fast"
    elif velocity < -0.2:
        return "fading"
    else:
        return "stable"


def _velocity_icon(velocity: float) -> str:
    """Get velocity indicator icon."""
    if velocity > 0.5:
        return "🔥"  # Rising fast
    elif velocity > 0.2:
        return "📈"  # Rising
    elif velocity < -0.5:
        return "📉"  # Fading fast
    elif velocity < -0.2:
        return "↘️"  # Fading
    else:
        return "➡️"  # Stable


def _save_trends_snapshot(topics: List[Tuple[str, float]]) -> None:
    """Save current trends for velocity calculation."""
    try:
        os.makedirs(os.path.dirname(TRENDS_HISTORY_FILE), exist_ok=True)
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "topics": [{"topic": t, "score": s} for t, s in topics]
        }
        with open(TRENDS_HISTORY_FILE, 'w') as f:
            json.dump(snapshot, f)
    except Exception:
        pass  # Silently fail on save errors


def _load_trends_snapshot() -> List[Dict]:
    """Load previous trends snapshot."""
    try:
        if os.path.exists(TRENDS_HISTORY_FILE):
            with open(TRENDS_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                
            # Check if snapshot is recent (within 6 hours)
            snapshot_time = date_parser.parse(data["timestamp"])
            if snapshot_time.tzinfo:
                snapshot_time = snapshot_time.replace(tzinfo=None)
            
            if datetime.now() - snapshot_time < timedelta(hours=6):
                return data.get("topics", [])
    except Exception:
        pass
    
    return []

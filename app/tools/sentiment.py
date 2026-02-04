"""AI Sentiment Intelligence V2 - Institutional analyst-style analysis."""
import os
import json
import re
import boto3
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


# V2 Institutional Analyst Prompt
SENTIMENT_PROMPT_V2 = """You are a senior financial intelligence analyst.

Analyze the sentiment and narrative tone of these news headlines.

Focus on:
• Market narrative
• Risk signals
• Momentum direction
• Opportunity vs threat balance
• Confidence based on coverage volume and consistency

Headlines:
{titles_json}

Return ONLY valid JSON in this exact format:
{{
  "overall": "positive" or "neutral" or "negative",
  "mood_label": "3-5 word narrative mood summary",
  "confidence": "high" or "medium" or "low",
  "direction": "improving" or "stable" or "deteriorating",
  "momentum_strength": "strong" or "moderate" or "weak",
  "risk_level": "low" or "moderate" or "high",
  "market_bias": "risk_on" or "balanced" or "risk_off",
  "reasoning": "2-3 sentence analyst explanation of narrative drivers",
  "positive_signals": ["signal 1", "signal 2"],
  "negative_signals": ["signal 1", "signal 2"],
  "emerging_themes": ["theme 1", "theme 2"],
  "score": 0.0-1.0,
  "breakdown": {{"positive": N, "neutral": N, "negative": N}}
}}

Rules:
- Think like Bloomberg or institutional analyst
- Avoid generic sentiment descriptions
- Focus on WHY sentiment exists
- Infer narrative momentum when possible"""


def analyze_sentiment(news_items: List[Dict]) -> Dict:
    """Generate institutional analyst-style sentiment intelligence."""
    if not news_items:
        return _empty_sentiment()
    
    if os.getenv("USE_MOCK_PLANNER", "true").lower() == "true":
        return _get_mock_sentiment_v2(news_items)
    
    try:
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
        titles = [item.get("title", "") for item in news_items[:12]]
        
        prompt = SENTIMENT_PROMPT_V2.format(titles_json=json.dumps(titles, indent=2))
        
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 800, "temperature": 0.4}
        }
        
        response = client.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        output_text = re.sub(r'```json\s*|```\s*', '', output_text)
        
        sentiment = json.loads(output_text.strip())
        return _validate_sentiment(sentiment)
        
    except Exception as e:
        print(f"Sentiment error: {e}")
        return _get_mock_sentiment_v2(news_items)


def _empty_sentiment() -> Dict:
    return {
        "overall": "neutral",
        "mood_label": "Insufficient data",
        "confidence": "low",
        "direction": "stable",
        "momentum_strength": "weak",
        "risk_level": "low",
        "market_bias": "balanced",
        "reasoning": "Insufficient data to analyze sentiment.",
        "positive_signals": [],
        "negative_signals": [],
        "emerging_themes": [],
        "score": 0.5,
        "breakdown": {"positive": 0, "neutral": 0, "negative": 0}
    }


def _validate_sentiment(sentiment: Dict) -> Dict:
    """Ensure all required fields exist."""
    defaults = _empty_sentiment()
    for key in defaults:
        if key not in sentiment:
            sentiment[key] = defaults[key]
    return sentiment


def _get_mock_sentiment_v2(news_items: List[Dict]) -> Dict:
    """Generate V2 institutional-style mock sentiment."""
    total = len(news_items)
    all_text = " ".join(item.get("title", "") for item in news_items).lower()
    
    # Keyword analysis
    positive_words = ["success", "growth", "launch", "new", "innovation", "surge", "gains", 
                      "deal", "partnership", "expansion", "record", "breakthrough", "optimism"]
    negative_words = ["fail", "crash", "lawsuit", "decline", "risk", "fear", "drop", 
                      "crisis", "concern", "uncertainty", "warning", "threat", "pressure"]
    momentum_words = ["momentum", "accelerat", "surge", "rally", "climb", "soar"]
    risk_words = ["regulat", "lawsuit", "investigation", "probe", "fine", "penalty"]
    
    pos_count = sum(1 for item in news_items if any(w in item.get("title", "").lower() for w in positive_words))
    neg_count = sum(1 for item in news_items if any(w in item.get("title", "").lower() for w in negative_words))
    neutral_count = total - pos_count - neg_count
    
    # Momentum analysis
    has_momentum = any(w in all_text for w in momentum_words)
    has_risk = any(w in all_text for w in risk_words)
    
    # Determine overall sentiment
    if pos_count > neg_count + 3:
        overall = "positive"
        mood = "Strong bullish momentum"
        direction = "improving"
        score = 0.75
        market_bias = "risk_on"
    elif pos_count > neg_count + 1:
        overall = "positive"
        mood = "Cautiously optimistic"
        direction = "improving" if has_momentum else "stable"
        score = 0.65
        market_bias = "risk_on" if has_momentum else "balanced"
    elif neg_count > pos_count + 3:
        overall = "negative"
        mood = "Risk-off sentiment prevailing"
        direction = "deteriorating"
        score = 0.25
        market_bias = "risk_off"
    elif neg_count > pos_count + 1:
        overall = "negative"
        mood = "Elevated caution signals"
        direction = "deteriorating" if has_risk else "stable"
        score = 0.35
        market_bias = "risk_off" if has_risk else "balanced"
    elif pos_count > neg_count:
        overall = "positive"
        mood = "Mild optimism with hedging"
        direction = "stable"
        score = 0.58
        market_bias = "balanced"
    elif neg_count > pos_count:
        overall = "negative"
        mood = "Slight bearish undertone"
        direction = "stable"
        score = 0.42
        market_bias = "balanced"
    else:
        overall = "neutral"
        mood = "Mixed signals, balanced narrative"
        direction = "stable"
        score = 0.5
        market_bias = "balanced"
    
    # Confidence based on data volume & consistency
    if total >= 10:
        confidence = "high"
        momentum_strength = "strong" if has_momentum else "moderate"
    elif total >= 5:
        confidence = "medium"
        momentum_strength = "moderate"
    else:
        confidence = "low"
        momentum_strength = "weak"
    
    # Risk assessment
    if neg_count > 4 or has_risk:
        risk_level = "high"
    elif neg_count > 2:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Generate reasoning
    reasoning = _generate_reasoning(pos_count, neg_count, total, overall, has_momentum, has_risk)
    
    # Extract signals
    positive_signals = _extract_signals(news_items, positive_words, 3)
    negative_signals = _extract_signals(news_items, negative_words, 3)
    
    # Emerging themes (extract proper nouns)
    themes = _extract_themes(news_items)
    
    return {
        "overall": overall,
        "mood_label": mood,
        "confidence": confidence,
        "direction": direction,
        "momentum_strength": momentum_strength,
        "risk_level": risk_level,
        "market_bias": market_bias,
        "reasoning": reasoning,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "emerging_themes": themes,
        "score": score,
        "breakdown": {
            "positive": pos_count,
            "neutral": neutral_count,
            "negative": neg_count
        }
    }


def _generate_reasoning(pos: int, neg: int, total: int, overall: str, momentum: bool, risk: bool) -> str:
    """Generate analyst-style reasoning."""
    if overall == "positive":
        base = f"Analysis of {total} headlines reveals predominantly constructive narrative. "
        if momentum:
            base += "Strong momentum indicators suggest accelerating positive sentiment. "
        if not risk:
            base += "Risk signals remain contained, supporting bullish thesis."
        else:
            base += "Some regulatory headwinds noted but manageable given overall tone."
    elif overall == "negative":
        base = f"Coverage analysis ({total} sources) indicates cautionary positioning. "
        if risk:
            base += "Elevated risk signals detected across multiple stories. "
        base += "Recommend monitoring for sentiment shift before increasing exposure."
    else:
        base = f"Balanced coverage across {total} headlines with no dominant directional bias. "
        base += "Market narrative remains mixed with both opportunity and risk factors in play."
    return base


def _extract_signals(items: List[Dict], keywords: List[str], limit: int) -> List[str]:
    """Extract signal descriptions from headlines."""
    signals = []
    for item in items:
        title = item.get("title", "")
        for word in keywords:
            if word in title.lower():
                # Create a clean signal name
                if word in ["deal", "partnership"]:
                    signals.append("Strategic partnerships/M&A activity")
                elif word in ["growth", "expansion"]:
                    signals.append("Growth & expansion coverage")
                elif word in ["launch", "new", "innovation"]:
                    signals.append("Product/innovation announcements")
                elif word in ["risk", "concern", "uncertainty"]:
                    signals.append("Risk/uncertainty mentions")
                elif word in ["crash", "decline", "drop"]:
                    signals.append("Market decline coverage")
                elif word in ["lawsuit", "regulat"]:
                    signals.append("Regulatory/legal concerns")
                break
    # Deduplicate and limit
    return list(dict.fromkeys(signals))[:limit]


def _extract_themes(items: List[Dict]) -> List[str]:
    """Extract emerging themes from headlines."""
    import re
    from collections import Counter
    
    proper_nouns = []
    for item in items:
        title = item.get("title", "")
        # Find capitalized words (proper nouns)
        found = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        proper_nouns.extend(found)
    
    # Filter common words
    stopwords = {"The", "This", "That", "New", "News", "Today", "Says", "Report", "Update"}
    filtered = [n for n in proper_nouns if n not in stopwords and len(n) > 3]
    
    # Get top themes
    top = Counter(filtered).most_common(4)
    return [theme for theme, count in top if count >= 2]

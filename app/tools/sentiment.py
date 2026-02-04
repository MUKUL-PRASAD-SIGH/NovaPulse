"""Sentiment analysis tool using Amazon Nova."""
import os
import json
import re
import boto3
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def analyze_sentiment(news_items: List[Dict]) -> Dict:
    """Analyze sentiment of news headlines."""
    if not news_items:
        return {"overall": "neutral", "score": 0.5, "breakdown": {}}
    
    if os.getenv("USE_MOCK_PLANNER", "true").lower() == "true":
        return _get_mock_sentiment(news_items)
    
    try:
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
        titles = [item.get("title", "") for item in news_items[:10]]
        
        prompt = f"""Analyze sentiment of these headlines.
Headlines: {json.dumps(titles)}
Return ONLY valid JSON: {{"overall": "positive/neutral/negative", "score": 0.0-1.0, "breakdown": {{"positive": N, "neutral": N, "negative": N}}}}"""
        
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.3}
        }
        
        response = client.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        output_text = re.sub(r'```json\s*|```\s*', '', output_text)
        return json.loads(output_text.strip())
        
    except Exception as e:
        print(f"Sentiment error: {e}")
        return _get_mock_sentiment(news_items)


def _get_mock_sentiment(news_items: List[Dict]) -> Dict:
    total = len(news_items)
    positive_words = ["success", "growth", "launch", "new", "innovation"]
    negative_words = ["fail", "crash", "lawsuit", "decline", "risk"]
    
    pos_count = neg_count = 0
    for item in news_items:
        title_lower = item.get("title", "").lower()
        if any(word in title_lower for word in positive_words):
            pos_count += 1
        elif any(word in title_lower for word in negative_words):
            neg_count += 1
    
    neutral_count = total - pos_count - neg_count
    overall = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
    score = 0.6 if pos_count > neg_count else 0.4 if neg_count > pos_count else 0.5
    
    return {"overall": overall, "score": score, "breakdown": {"positive": pos_count, "neutral": neutral_count, "negative": neg_count}}

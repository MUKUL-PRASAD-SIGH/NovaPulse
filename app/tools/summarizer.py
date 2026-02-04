"""AI-powered news summarizer using Amazon Nova."""
import os
import json
import re
import boto3
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def summarize_news(news_items: List[Dict]) -> Dict:
    """Generate AI summary of news articles."""
    if not news_items:
        return {"summary": "No news to summarize.", "key_points": []}
    
    if os.getenv("USE_MOCK_PLANNER", "true").lower() == "true":
        return _get_mock_summary(news_items)
    
    try:
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
        titles = [item.get("title", "") for item in news_items[:10]]
        
        prompt = f"""Summarize these news headlines in 3 sentences. Extract 3 key themes.
Headlines: {json.dumps(titles)}
Return ONLY valid JSON: {{"summary": "...", "key_points": ["...", "...", "..."]}}"""
        
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 400, "temperature": 0.7}
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
        print(f"Summarizer error: {e}")
        return _get_mock_summary(news_items)


def _get_mock_summary(news_items: List[Dict]) -> Dict:
    titles = [item.get("title", "")[:60] for item in news_items[:3]]
    return {
        "summary": f"Today's news covers {len(news_items)} stories focusing on recent developments. Key headlines include updates on technology advancements and industry trends.",
        "key_points": titles if titles else ["No headlines available"]
    }

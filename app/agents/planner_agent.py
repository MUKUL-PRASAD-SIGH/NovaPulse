"""Planner Agent using Amazon Nova."""
import os
import json
import boto3
import re
from typing import Dict
from dotenv import load_dotenv

from app.core.plan_validator import validate_plan
from app.core.tool_registry import get_tool_descriptions
from app.memory.store import log

load_dotenv()


PLANNER_PROMPT = """You are Nova Intelligence Agent - an AI news analysis system with Multi-Agent capabilities.

Convert user requests into a JSON task plan.

AVAILABLE TOOLS:
{tool_descriptions}

USER REQUEST: {user_input}

RESPOND WITH ONLY VALID JSON (no markdown, no explanation):
{{
    "intent": "describe the goal briefly",
    "domain": "the topic user asked about",
    "steps": [
        {{"tool": "news_fetcher", "params": {{"topic": "extract topic from user request", "sources": ["google"], "limit": 5}}}},
        {{"tool": "exporter", "params": {{"filename": "report", "format": "json"}}}}
    ]
}}

RULES:
1. Extract the ACTUAL topic from user request (e.g., "moltbot", "openai", "tesla")
2. Add summarizer if user wants summary
3. Add sentiment if user asks about sentiment/mood/tone
4. Add trends if user asks about trending/popular topics
5. Add web_scraper if user wants full article content or deep analysis
6. Add entity_extractor if user wants to know about people, organizations, or locations
7. Add image_analyzer if user mentions images or visual content
8. Add social_monitor if user wants social media buzz or Reddit/Twitter data
9. Add research_assistant if user wants academic papers, GitHub repos, or technical research
10. Always end with exporter"""


def get_mock_plan(user_input: str) -> Dict:
    """Generate mock plan - extracts topic from input."""
    input_lower = user_input.lower()
    
    # Extract topic - find words that aren't common
    common = {"get", "latest", "news", "about", "show", "me", "find", "the", "with", "and", "of", "for", "on", "a", "an"}
    words = re.findall(r'\b[a-z]+\b', input_lower)
    topic = "ai"
    for word in words:
        if word not in common and len(word) > 2:
            topic = word
            break
    
    steps = [
        {"tool": "news_fetcher", "params": {"topic": topic, "sources": ["google"], "limit": 5}}
    ]
    
    # Core intelligence tools
    if any(w in input_lower for w in ["summar", "digest", "brief"]):
        steps.append({"tool": "summarizer", "params": {}})
    if any(w in input_lower for w in ["sentiment", "mood", "tone"]):
        steps.append({"tool": "sentiment", "params": {}})
    if any(w in input_lower for w in ["trend", "trending", "popular"]):
        steps.append({"tool": "trends", "params": {}})
    
    # MAS tools
    if any(w in input_lower for w in ["full", "article", "content", "scrape", "deep", "detailed"]):
        steps.append({"tool": "web_scraper", "params": {}})
    if any(w in input_lower for w in ["entity", "entities", "people", "person", "organization", "company", "location"]):
        steps.append({"tool": "entity_extractor", "params": {}})
    if any(w in input_lower for w in ["image", "images", "photo", "picture", "visual"]):
        steps.append({"tool": "image_analyzer", "params": {}})
    if any(w in input_lower for w in ["social", "reddit", "twitter", "buzz", "discussion"]):
        steps.append({"tool": "social_monitor", "params": {"topic": topic, "platforms": ["reddit"]}})
    if any(w in input_lower for w in ["research", "paper", "academic", "github", "repo", "stackoverflow", "technical"]):
        steps.append({"tool": "research_assistant", "params": {"query": topic}})
    
    export_format = "markdown" if "markdown" in input_lower else "csv" if "csv" in input_lower else "json"
    steps.append({"tool": "exporter", "params": {"filename": f"{topic}_report", "format": export_format}})
    
    return {"intent": f"Get {topic} news", "domain": topic, "steps": steps}


def plan_task(user_input: str) -> Dict:
    """Convert user input to task plan."""
    log("INFO", f"Planning task for: {user_input}")
    
    if os.getenv("USE_MOCK_PLANNER", "true").lower() == "true":
        plan = get_mock_plan(user_input)
        log("INFO", "Using mock planner", {"plan": plan})
        return plan
    
    try:
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
        
        prompt = PLANNER_PROMPT.format(
            tool_descriptions=get_tool_descriptions(),
            user_input=user_input
        )
        
        # Nova uses messages format
        body = {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": 500,
                "temperature": 0.3
            }
        }
        
        response = client.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        # Extract text from Nova response
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        
        # Clean up - remove markdown code blocks if present
        output_text = re.sub(r'```json\s*', '', output_text)
        output_text = re.sub(r'```\s*', '', output_text)
        
        plan = json.loads(output_text.strip())
        
        # Clean the plan (remove empty data params)
        plan = _clean_plan(plan)
        
        is_valid, error_msg = validate_plan(plan)
        if not is_valid:
            log("WARNING", f"Invalid plan: {error_msg}")
            return get_mock_plan(user_input)
        
        log("INFO", "Nova plan generated", {"plan": plan})
        return plan
        
    except Exception as e:
        log("ERROR", f"Nova planner error: {e}")
        return get_mock_plan(user_input)


def _clean_plan(plan: Dict) -> Dict:
    """Remove empty/useless params from plan steps."""
    if "steps" in plan:
        for step in plan["steps"]:
            if step.get("tool") == "exporter" and "params" in step:
                # Remove data param - executor will fill it from context
                # Nova often puts garbage like "", "news_items", etc.
                if "data" in step["params"]:
                    del step["params"]["data"]
    return plan

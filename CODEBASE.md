# Nova Intelligence Agent - Complete Codebase

> **Last Updated:** 2026-02-04  
> **Share this file to provide full context to other AI tools.**

---

## 📁 Project Structure

```
NovaAI/
├── app/
│   ├── main.py
│   ├── agents/ (planner_agent.py, executor_agent.py)
│   ├── tools/ (multi_fetcher.py, summarizer.py, sentiment.py, trends.py, exporter.py)
│   ├── api/ (routes.py)
│   ├── core/ (tool_registry.py)
│   └── memory/ (store.py)
├── frontend/ (index.html, app.js, style.css)
└── requirements.txt
```

---

## app/main.py

```python
"""Nova Intelligence Agent - Main Application Entry Point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api.routes import router

app = FastAPI(
    title="Nova Intelligence Agent",
    description="AI-powered news intelligence with voice interface",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["Intelligence"])

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.on_event("startup")
async def startup_event():
    print("🧠 Nova Intelligence Agent starting...")
    print("📡 API available at: http://localhost:8000/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

## app/agents/planner_agent.py

```python
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

PLANNER_PROMPT = """You are Nova Intelligence Agent - an AI news analysis system.

Convert user requests into a JSON task plan.

AVAILABLE TOOLS:
{tool_descriptions}

USER REQUEST: {user_input}

RESPOND WITH ONLY VALID JSON (no markdown):
{{
    "intent": "describe the goal briefly",
    "domain": "the topic user asked about",
    "steps": [
        {{"tool": "news_fetcher", "params": {{"topic": "extract topic", "limit": 5}}}},
        {{"tool": "exporter", "params": {{"filename": "report", "format": "json"}}}}
    ]
}}

RULES:
1. Extract ACTUAL topic from user request
2. Add summarizer if user wants summary
3. Add sentiment if user asks about sentiment/mood/tone
4. Add trends if user asks about trending topics
5. Always end with exporter"""

def get_mock_plan(user_input: str) -> Dict:
    """Generate mock plan - extracts topic from input."""
    input_lower = user_input.lower()
    common = {"get", "latest", "news", "about", "show", "me", "find", "the", "with", "and", "of", "for"}
    words = re.findall(r'\b[a-z]+\b', input_lower)
    topic = "ai"
    for word in words:
        if word not in common and len(word) > 2:
            topic = word
            break
    
    steps = [{"tool": "news_fetcher", "params": {"topic": topic, "limit": 5}}]
    
    if any(w in input_lower for w in ["summar", "digest", "brief"]):
        steps.append({"tool": "summarizer", "params": {}})
    if any(w in input_lower for w in ["sentiment", "mood", "tone"]):
        steps.append({"tool": "sentiment", "params": {}})
    if any(w in input_lower for w in ["trend", "trending", "popular"]):
        steps.append({"tool": "trends", "params": {}})
    
    export_format = "markdown" if "markdown" in input_lower else "json"
    steps.append({"tool": "exporter", "params": {"filename": f"{topic}_report", "format": export_format}})
    
    return {"intent": f"Get {topic} news", "domain": topic, "steps": steps}

def plan_task(user_input: str) -> Dict:
    """Convert user input to task plan."""
    log("INFO", f"Planning task for: {user_input}")
    
    if os.getenv("USE_MOCK_PLANNER", "true").lower() == "true":
        return get_mock_plan(user_input)
    
    try:
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
        prompt = PLANNER_PROMPT.format(tool_descriptions=get_tool_descriptions(), user_input=user_input)
        
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 500, "temperature": 0.3}
        }
        
        response = client.invoke_model(modelId='amazon.nova-lite-v1:0', body=json.dumps(body), contentType='application/json')
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        output_text = re.sub(r'```json\s*|```\s*', '', output_text)
        
        plan = json.loads(output_text.strip())
        plan = _clean_plan(plan)
        
        is_valid, error_msg = validate_plan(plan)
        if not is_valid:
            return get_mock_plan(user_input)
        
        log("INFO", "Nova plan generated", {"plan": plan})
        return plan
        
    except Exception as e:
        log("ERROR", f"Nova planner error: {e}")
        return get_mock_plan(user_input)

def _clean_plan(plan: Dict) -> Dict:
    """Remove empty params from plan steps."""
    if "steps" in plan:
        for step in plan["steps"]:
            if step.get("tool") == "exporter" and "params" in step:
                if "data" in step["params"]:
                    del step["params"]["data"]
    return plan
```

---

## app/agents/executor_agent.py

```python
"""Executor Agent - runs tools in sequence with context passing."""
from typing import Dict, Any, List
from app.core.tool_registry import get_tool, list_tools
from app.models.schemas import TaskPlan, ToolStep
from app.memory.store import save_result, log

def execute_plan(plan: TaskPlan) -> Dict[str, Any]:
    """Execute a task plan by running each tool step."""
    log("INFO", f"Executing plan: {plan.intent}", {"steps": len(plan.steps)})
    
    result = {
        "intent": plan.intent,
        "domain": plan.domain,
        "tools_executed": [],
        "data": {},
        "errors": [],
        "success": True
    }
    
    context: Dict[str, Any] = {}
    
    for i, step in enumerate(plan.steps):
        tool_result = _execute_step(step, context, i)
        
        if tool_result["success"]:
            result["tools_executed"].append({"tool": step.tool, "success": True})
            _update_context(context, step.tool, tool_result["output"])
        else:
            result["errors"].append(tool_result["error"])
            result["tools_executed"].append({"tool": step.tool, "success": False, "error": tool_result["error"]})
    
    result["data"] = context
    result["success"] = len(result["errors"]) == 0
    save_result(result)
    log("INFO", f"Execution complete: {len(result['tools_executed'])} tools run")
    return result

def _execute_step(step: ToolStep, context: Dict, step_index: int) -> Dict:
    """Execute a single tool step."""
    tool_fn = get_tool(step.tool)
    if not tool_fn:
        return {"success": False, "error": f"Tool not found: {step.tool}", "output": None}
    
    try:
        params = dict(step.params)
        params = _inject_context(step.tool, params, context)
        log("DEBUG", f"Running tool: {step.tool}", {"params": list(params.keys())})
        output = tool_fn(**params)
        return {"success": True, "output": output, "error": None}
    except Exception as e:
        log("ERROR", f"Tool {step.tool} failed: {str(e)}")
        return {"success": False, "error": f"{step.tool}: {str(e)}", "output": None}

def _inject_context(tool_name: str, params: Dict, context: Dict) -> Dict:
    """Inject context data into tool parameters."""
    if tool_name in ["summarizer", "sentiment", "trends"]:
        if "news" in context:
            params["news_items"] = context["news"]
    
    if tool_name == "exporter":
        params["data"] = {
            "news": context.get("news", []),
            "summary": context.get("summary"),
            "sentiment": context.get("sentiment"),
            "trends": context.get("trends")
        }
        params["data"] = {k: v for k, v in params["data"].items() if v is not None}
    
    return params

def _update_context(context: Dict, tool_name: str, output: Any):
    """Update context with tool output."""
    if tool_name == "news_fetcher":
        context["news"] = output
    elif tool_name == "summarizer":
        context["summary"] = output
    elif tool_name == "sentiment":
        context["sentiment"] = output
    elif tool_name == "trends":
        context["trends"] = output
    elif tool_name == "exporter":
        context["exported_file"] = output
```

---

## app/tools/multi_fetcher.py

```python
"""Multi-source news fetcher - Tavily, GNews, RSS in parallel."""
import feedparser
import os
import httpx
from typing import List, Dict
from difflib import SequenceMatcher
from dotenv import load_dotenv
from app.memory.store import log

load_dotenv()

def fetch_news_multi(topic: str, limit: int = 5, **kwargs) -> List[Dict]:
    """Fetch from multiple sources with failover."""
    log("INFO", f"Multi-source fetch for: {topic}")
    
    all_articles = []
    
    # 1. RSS (always free)
    rss_result = _fetch_rss_sync(topic, limit)
    if rss_result["success"]:
        all_articles.extend(rss_result["articles"])
    
    # 2. GNews
    gnews_key = os.getenv("GNEWS_API_KEY", "")
    if gnews_key:
        gnews_result = _fetch_gnews_sync(topic, limit, gnews_key)
        if gnews_result["success"]:
            all_articles.extend(gnews_result["articles"])
    
    # 3. Tavily
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        tavily_result = _fetch_tavily_sync(topic, limit, tavily_key)
        if tavily_result["success"]:
            all_articles.extend(tavily_result["articles"])
    
    unique = _deduplicate(all_articles)
    log("INFO", f"Multi-source complete: {len(unique)} articles")
    return unique

def _fetch_rss_sync(topic: str, limit: int) -> Dict:
    try:
        url = f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = [{"title": e.get("title"), "link": e.get("link"), "source": "rss", "published": e.get("published", "")} for e in feed.entries[:limit]]
        return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"RSS failed: {e}")
        return {"success": False, "articles": []}

def _fetch_gnews_sync(topic: str, limit: int, api_key: str) -> Dict:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get("https://gnews.io/api/v4/search", params={"apikey": api_key, "q": topic, "lang": "en", "max": limit})
            if resp.status_code in [429, 403]:
                return {"success": False, "articles": []}
            resp.raise_for_status()
            data = resp.json()
            articles = [{"title": i.get("title"), "link": i.get("url"), "source": "gnews", "published": i.get("publishedAt", "")} for i in data.get("articles", [])[:limit]]
            return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"GNews failed: {e}")
        return {"success": False, "articles": []}

def _fetch_tavily_sync(topic: str, limit: int, api_key: str) -> Dict:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post("https://api.tavily.com/search", json={
                "api_key": api_key,
                "query": f"{topic} breaking news today",
                "search_depth": "advanced",
                "max_results": limit + 5,
                "include_domains": ["reuters.com", "bloomberg.com", "cnbc.com", "bbc.com", "cnn.com"]
            })
            if resp.status_code == 429:
                return {"success": False, "articles": []}
            resp.raise_for_status()
            data = resp.json()
            articles = []
            skip_titles = ["latest stories", "today's latest", "latest news"]
            for r in data.get("results", []):
                if len(articles) >= limit:
                    break
                title = _clean_title(r.get("title", ""))
                if title and len(title) >= 15 and not any(s in title.lower() for s in skip_titles):
                    articles.append({"title": title, "link": r.get("url"), "source": "tavily", "published": ""})
            return {"success": True, "articles": articles}
    except Exception as e:
        log("ERROR", f"Tavily failed: {e}")
        return {"success": False, "articles": []}

def _clean_title(title: str) -> str:
    separators = [' | ', ' - ', ' – ', ' — ']
    for sep in separators:
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if len(p.strip()) >= 15]
            if parts:
                return max(parts, key=len)
    return title.strip()

def _deduplicate(articles: List[Dict]) -> List[Dict]:
    seen_urls, seen_titles, unique = set(), [], []
    for a in articles:
        url, title = a.get("link", ""), a.get("title", "")
        if url in seen_urls:
            continue
        if any(SequenceMatcher(None, title.lower(), t.lower()).ratio() > 0.85 for t in seen_titles):
            continue
        seen_urls.add(url)
        seen_titles.append(title)
        unique.append(a)
    return unique
```

---

## app/tools/summarizer.py

```python
"""AI-powered news summarizer using Amazon Nova."""
import os, json, re, boto3
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def summarize_news(news_items: List[Dict]) -> Dict:
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
        
        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}], "inferenceConfig": {"maxTokens": 400, "temperature": 0.7}}
        response = client.invoke_model(modelId='amazon.nova-lite-v1:0', body=json.dumps(body), contentType='application/json')
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        output_text = re.sub(r'```json\s*|```\s*', '', output_text)
        return json.loads(output_text.strip())
    except Exception as e:
        return _get_mock_summary(news_items)

def _get_mock_summary(news_items: List[Dict]) -> Dict:
    titles = [item.get("title", "")[:60] for item in news_items[:3]]
    return {"summary": f"Today's news covers {len(news_items)} stories.", "key_points": titles}
```

---

## app/tools/sentiment.py

```python
"""Sentiment analysis using Amazon Nova."""
import os, json, re, boto3
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def analyze_sentiment(news_items: List[Dict]) -> Dict:
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
        
        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}], "inferenceConfig": {"maxTokens": 200, "temperature": 0.3}}
        response = client.invoke_model(modelId='amazon.nova-lite-v1:0', body=json.dumps(body), contentType='application/json')
        result = json.loads(response['body'].read())
        output_text = result.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '{}')
        return json.loads(re.sub(r'```json\s*|```\s*', '', output_text).strip())
    except Exception as e:
        return _get_mock_sentiment(news_items)

def _get_mock_sentiment(news_items: List[Dict]) -> Dict:
    total = len(news_items)
    pos_words = ["success", "growth", "launch", "new", "innovation"]
    neg_words = ["fail", "crash", "lawsuit", "decline", "risk"]
    pos_count = neg_count = 0
    for item in news_items:
        title = item.get("title", "").lower()
        if any(w in title for w in pos_words): pos_count += 1
        elif any(w in title for w in neg_words): neg_count += 1
    neutral_count = total - pos_count - neg_count
    overall = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
    score = 0.6 if pos_count > neg_count else 0.4 if neg_count > pos_count else 0.5
    return {"overall": overall, "score": score, "breakdown": {"positive": pos_count, "neutral": neutral_count, "negative": neg_count}}
```

---

## app/tools/trends.py

```python
"""Trend extraction from headlines."""
import re
from collections import Counter
from typing import List, Dict

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were"}

def extract_trends(news_items: List[Dict]) -> Dict:
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
    return {"trending_topics": [{"topic": word, "mentions": count} for word, count in top_topics], "total_articles": len(news_items)}
```

---

## app/tools/exporter.py

```python
"""Multi-format data exporter - JSON, Markdown, CSV."""
import json, os, csv
from datetime import datetime
from typing import Dict, Any

OUTPUT_DIR = "output"

def export_data(data: Dict[str, Any], filename: str = "report", format: str = "json") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "markdown":
        return _export_markdown(data, filename, timestamp)
    elif format == "csv":
        return _export_csv(data, filename, timestamp)
    else:
        return _export_json(data, filename, timestamp)

def _export_json(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.json"
    output = {"generated_at": datetime.now().isoformat(), **data}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path

def _export_markdown(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.md"
    lines = ["# Nova Intelligence Report", f"\n*Generated: {timestamp}*\n"]
    if "summary" in data:
        s = data["summary"]
        lines.append("## Summary")
        lines.append(s.get("summary", str(s)) if isinstance(s, dict) else str(s))
    if "news" in data:
        lines.append("\n## Articles")
        for item in data["news"]:
            lines.append(f"- [{item.get('title')}]({item.get('link')})")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path

def _export_csv(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.csv"
    news = data.get("news", [{"title": "No data", "link": "", "source": ""}])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "link", "source"], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(news)
    return path
```

---

## app/api/routes.py

```python
"""FastAPI routes for Nova Intelligence Agent."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.models.schemas import CommandRequest, TaskPlan
from app.agents.planner_agent import plan_task
from app.agents.executor_agent import execute_plan
from app.memory.store import save_plan, get_recent_plans, get_recent_results
from app.core.tool_registry import list_tools

router = APIRouter()

@router.post("/command")
async def process_command(request: CommandRequest) -> Dict[str, Any]:
    try:
        plan_dict = plan_task(request.text)
        save_plan(plan_dict, request.text)
        plan = TaskPlan(**plan_dict)
        result = execute_plan(plan)
        return {"success": True, "plan": plan_dict, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    return {
        "name": "Nova Intelligence Agent",
        "version": "1.0",
        "features": [
            {"id": "multi_source", "name": "Multi-Source News", "icon": "📰"},
            {"id": "ai_summary", "name": "AI Summary", "icon": "🧠"},
            {"id": "sentiment", "name": "Sentiment Analysis", "icon": "💭"},
            {"id": "trends", "name": "Trend Detection", "icon": "📊"},
            {"id": "export", "name": "Multi-Format Export", "icon": "💾"},
        ],
        "tools": list_tools()
    }

@router.get("/history")
async def get_history() -> Dict[str, Any]:
    return {"plans": get_recent_plans(10), "results": get_recent_results(10)}

@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "agent": "Nova Intelligence Agent"}
```

---

## app/core/tool_registry.py

```python
"""Tool registry for Nova Intelligence Agent."""
from typing import Callable, Dict, List, Optional
from app.tools.multi_fetcher import fetch_news_multi
from app.tools.summarizer import summarize_news
from app.tools.sentiment import analyze_sentiment
from app.tools.trends import extract_trends
from app.tools.exporter import export_data

TOOLS: Dict[str, Callable] = {
    "news_fetcher": fetch_news_multi,
    "summarizer": summarize_news,
    "sentiment": analyze_sentiment,
    "trends": extract_trends,
    "exporter": export_data,
}

TOOL_DESCRIPTIONS = {
    "news_fetcher": "Fetches news from multiple sources (Tavily, GNews, RSS). Params: topic, limit",
    "summarizer": "AI summary of news. Params: news_items",
    "sentiment": "Sentiment analysis. Params: news_items",
    "trends": "Extract trending topics. Params: news_items",
    "exporter": "Save to file. Params: data, filename, format",
}

def get_tool(name: str) -> Optional[Callable]:
    return TOOLS.get(name)

def list_tools() -> List[str]:
    return list(TOOLS.keys())

def get_tool_descriptions() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
```

---

## requirements.txt

```
boto3>=1.34.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.26.0
python-dotenv>=1.0.0
pydantic>=2.5.0
feedparser>=6.0.10
nest_asyncio>=1.5.8
```

---

## .env.example

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
TAVILY_API_KEY=your_tavily_key
GNEWS_API_KEY=your_gnews_key
USE_MOCK_PLANNER=false
```

---

**End of Codebase Export**

"""Executor Agent for Nova Intelligence Agent.

Executes task plans by running tools in sequence.
Features:
- Dependency-aware execution (skip if dependency failed)
- Retry logic with exponential backoff
- Alternate tool fallback
- Auto step regeneration with parameter variation
- Dynamic plan rewriting on critical failures
- Per-step error isolation
- Context passing between tools
"""
from typing import Dict, Any, List, Optional
import time
import copy
import asyncio
import inspect
from app.core.tool_registry import get_tool, list_tools
from app.models.schemas import TaskPlan, ToolStep
from app.memory.store import save_result, log


# Dependency graph: which tools depend on which
TOOL_DEPENDENCIES = {
    "summarizer": ["news_fetcher"],
    "sentiment": ["news_fetcher"],
    "trends": ["news_fetcher"],
    # MAS tools
    "web_scraper": ["news_fetcher"],  # Needs URLs from news
    "entity_extractor": ["news_fetcher"],  # Needs text from news
    "image_analyzer": ["news_fetcher"],  # Needs image URLs from news
    "social_monitor": [],  # Independent
    "research_assistant": [],  # Independent
    "exporter": []  # Exporter can run with partial data
}

# Alternate tool fallbacks: if primary fails, try alternate
TOOL_FALLBACKS = {
    "news_fetcher": None,  # No fallback, it handles multi-source internally
    "summarizer": "_fallback_summarizer",
    "sentiment": "_fallback_sentiment",
    "trends": None,
}

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY_BASE = 1.0  # seconds

# Enable self-healing features
ENABLE_FALLBACKS = True
ENABLE_AUTO_REGENERATION = True
ENABLE_DYNAMIC_REWRITE = True


def execute_plan(plan: TaskPlan) -> Dict[str, Any]:
    """
    Execute a task plan with full self-healing capabilities.
    
    Features:
    - Dependency checking (skip if dependency failed)
    - Retry with exponential backoff
    - Alternate tool fallback
    - Auto step regeneration
    - Dynamic plan rewriting
    
    NEVER raises - always returns a result dict.
    """
    try:
        return _execute_plan_inner(plan)
    except Exception as e:
        print(f"[EXECUTOR ERROR] Top-level failure: {e}")
        return {
            "intent": getattr(plan, 'intent', 'unknown'),
            "domain": getattr(plan, 'domain', 'unknown'),
            "tools_executed": [],
            "data": {},
            "errors": [f"Executor crashed: {str(e)}"],
            "skipped": [],
            "fallbacks_used": [],
            "regenerated": [],
            "success": False
        }


def _execute_plan_inner(plan: TaskPlan) -> Dict[str, Any]:
    """Inner execution logic."""
    log("INFO", f"Executing plan: {plan.intent}", {"steps": len(plan.steps)})
    
    result = {
        "intent": plan.intent,
        "domain": plan.domain,
        "tools_executed": [],
        "data": {},
        "errors": [],
        "skipped": [],
        "fallbacks_used": [],
        "regenerated": [],
        "success": True
    }
    
    # Context for passing data between tools
    context: Dict[str, Any] = {}
    
    # Track which tools succeeded/failed
    tool_status: Dict[str, bool] = {}
    
    # Critical failure tracking for dynamic rewrite
    critical_failures = []
    
    for i, step in enumerate(plan.steps):
        # Check dependencies before execution
        skip_reason = _check_dependencies(step.tool, tool_status)
        
        if skip_reason:
            log("WARN", f"Skipping {step.tool}: {skip_reason}")
            result["skipped"].append({
                "tool": step.tool,
                "reason": skip_reason
            })
            tool_status[step.tool] = False
            continue
        
        # Execute with full self-healing pipeline
        tool_result = _execute_with_healing(step, context, i, result)
        
        if tool_result["success"]:
            result["tools_executed"].append({
                "tool": step.tool,
                "success": True,
                "retries": tool_result.get("retries", 0),
                "used_fallback": tool_result.get("used_fallback", False),
                "regenerated": tool_result.get("regenerated", False)
            })
            _update_context(context, step.tool, tool_result["output"])
            tool_status[step.tool] = True
        else:
            result["errors"].append(tool_result["error"])
            result["tools_executed"].append({
                "tool": step.tool,
                "success": False,
                "error": tool_result["error"],
                "retries": tool_result.get("retries", 0)
            })
            tool_status[step.tool] = False
            
            # Track critical failures for potential rewrite
            if step.tool == "news_fetcher":
                critical_failures.append(step.tool)
    
    # Dynamic plan rewriting: if news failed, try regenerating the plan
    if ENABLE_DYNAMIC_REWRITE and "news_fetcher" in critical_failures and not context.get("news"):
        log("WARN", "Critical failure detected - attempting dynamic plan recovery")
        result["dynamic_rewrite_attempted"] = True
        # Instead of full rewrite, add empty news fallback
        context["news"] = []
        result["data"]["recovery_note"] = "News fetch failed - partial results returned"
    
    # Sentiment × Trend Fusion: enhance trends with narrative signals
    if context.get("trends") and context.get("sentiment"):
        try:
            from app.tools.trends import fuse_trends_with_sentiment
            context["trends"] = fuse_trends_with_sentiment(context["trends"], context["sentiment"])
            log("INFO", "Sentiment-Trend fusion applied")
        except Exception as e:
            log("WARN", f"Fusion skipped: {e}")
    
    # Set final data from context
    result["data"] = context
    result["success"] = len(result["errors"]) == 0 or len(context.get("news", [])) > 0
    
    # Save to memory
    try:
        save_result(result)
    except Exception as e:
        print(f"[EXECUTOR WARN] Failed to save result: {e}")
    
    executed = len(result['tools_executed'])
    skipped = len(result['skipped'])
    fallbacks = len(result['fallbacks_used'])
    log("INFO", f"Execution complete: {executed} tools run, {skipped} skipped, {fallbacks} fallbacks used")
    
    return result


def _execute_with_healing(step: ToolStep, context: Dict, step_index: int, result: Dict) -> Dict:
    """Execute a step with all self-healing mechanisms."""
    
    # Step 1: Try primary execution with retry
    tool_result = _execute_step_with_retry(step, context, step_index)
    
    if tool_result["success"]:
        return tool_result
    
    # Step 2: Try alternate tool fallback
    if ENABLE_FALLBACKS:
        fallback_result = _try_fallback(step, context, step_index)
        if fallback_result and fallback_result["success"]:
            result["fallbacks_used"].append({
                "original": step.tool,
                "fallback": TOOL_FALLBACKS.get(step.tool)
            })
            fallback_result["used_fallback"] = True
            return fallback_result
    
    # Step 3: Try auto step regeneration with varied params
    if ENABLE_AUTO_REGENERATION:
        regen_result = _try_regeneration(step, context, step_index)
        if regen_result and regen_result["success"]:
            result["regenerated"].append(step.tool)
            regen_result["regenerated"] = True
            return regen_result
    
    # All healing attempts failed
    return tool_result


def _try_fallback(step: ToolStep, context: Dict, step_index: int) -> Optional[Dict]:
    """Try alternate tool if primary failed."""
    fallback_name = TOOL_FALLBACKS.get(step.tool)
    
    if not fallback_name:
        return None
    
    log("INFO", f"Attempting fallback for {step.tool}: {fallback_name}")
    
    # Use internal fallback functions
    if fallback_name == "_fallback_summarizer":
        return _fallback_summarizer(context)
    elif fallback_name == "_fallback_sentiment":
        return _fallback_sentiment(context)
    
    return None


def _fallback_summarizer(context: Dict) -> Dict:
    """Fallback summarizer - creates basic summary from headlines."""
    news = context.get("news", [])
    
    if not news:
        return {"success": False, "error": "No news to summarize", "output": None}
    
    # Create simple headline-based summary
    headlines = [item.get("title", "") for item in news[:5]]
    summary = {
        "summary": f"Key developments: {'; '.join(headlines[:3])}...",
        "key_points": headlines[:3],
        "fallback": True
    }
    
    log("INFO", "Fallback summarizer generated basic summary")
    return {"success": True, "output": summary, "error": None}


def _fallback_sentiment(context: Dict) -> Dict:
    """Fallback sentiment - basic keyword analysis."""
    news = context.get("news", [])
    
    if not news:
        return {"success": False, "error": "No news for sentiment", "output": None}
    
    # Simple keyword counting
    text = " ".join([item.get("title", "") for item in news])
    text_lower = text.lower()
    
    positive_words = ["gain", "rise", "growth", "up", "surge", "rally", "profit", "success"]
    negative_words = ["fall", "drop", "loss", "down", "crash", "fail", "decline", "crisis"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        overall = "positive"
        score = 0.6
    elif neg_count > pos_count:
        overall = "negative"
        score = 0.4
    else:
        overall = "neutral"
        score = 0.5
    
    sentiment = {
        "overall": overall,
        "score": score,
        "confidence": "low",
        "mood_label": f"Basic {overall} sentiment",
        "reasoning": "Fallback keyword-based analysis",
        "fallback": True,
        "breakdown": {"positive": pos_count, "neutral": len(news) - pos_count - neg_count, "negative": neg_count}
    }
    
    log("INFO", f"Fallback sentiment: {overall} (score: {score})")
    return {"success": True, "output": sentiment, "error": None}


def _try_regeneration(step: ToolStep, context: Dict, step_index: int) -> Optional[Dict]:
    """Try regenerating step with varied parameters."""
    
    # Only regenerate certain tools
    if step.tool not in ["news_fetcher", "summarizer"]:
        return None
    
    log("INFO", f"Attempting step regeneration for {step.tool}")
    
    # Create modified step with reduced parameters
    modified_step = copy.deepcopy(step)
    
    if step.tool == "news_fetcher":
        # Try with reduced article limit
        modified_step.params["limit"] = 3
    elif step.tool == "summarizer":
        # Try with fewer articles
        if "news" in context and len(context["news"]) > 3:
            context["news"] = context["news"][:3]
    
    # Execute modified step
    return _execute_step(modified_step, context, step_index)


def _check_dependencies(tool_name: str, tool_status: Dict[str, bool]) -> Optional[str]:
    """Check if tool's dependencies have succeeded."""
    dependencies = TOOL_DEPENDENCIES.get(tool_name, [])
    
    for dep in dependencies:
        if dep in tool_status and not tool_status[dep]:
            return f"dependency '{dep}' failed"
    
    return None


def _execute_step_with_retry(step: ToolStep, context: Dict, step_index: int) -> Dict:
    """Execute a tool step with retry logic and exponential backoff."""
    last_error = None
    
    for attempt in range(MAX_RETRIES + 1):
        result = _execute_step(step, context, step_index)
        
        if result["success"]:
            result["retries"] = attempt
            return result
        
        last_error = result["error"]
        
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY_BASE * (2 ** attempt)
            log("WARN", f"Retry {attempt + 1}/{MAX_RETRIES} for {step.tool} in {delay}s")
            time.sleep(delay)
    
    return {
        "success": False,
        "error": last_error,
        "output": None,
        "retries": MAX_RETRIES
    }


def _execute_step(step: ToolStep, context: Dict, step_index: int) -> Dict:
    """Execute a single tool step (handles both sync and async tools)."""
    tool_fn = get_tool(step.tool)
    
    if not tool_fn:
        return {
            "success": False,
            "error": f"Tool not found: {step.tool}",
            "output": None
        }
    
    try:
        params = dict(step.params)
        params = _inject_context(step.tool, params, context)
        
        log("DEBUG", f"Running tool: {step.tool}", {"params": list(params.keys())})
        
        # Handle async tools (MAS tools are async)
        if inspect.iscoroutinefunction(tool_fn):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context (FastAPI), use a new thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        output = pool.submit(asyncio.run, tool_fn(**params)).result(timeout=30)
                else:
                    output = loop.run_until_complete(tool_fn(**params))
            except RuntimeError:
                # No event loop, create one
                output = asyncio.run(tool_fn(**params))
        else:
            output = tool_fn(**params)
        
        return {
            "success": True,
            "output": output,
            "error": None
        }
        
    except Exception as e:
        log("ERROR", f"Tool {step.tool} failed: {str(e)}")
        return {
            "success": False,
            "error": f"{step.tool}: {str(e)}",
            "output": None
        }


def _inject_context(tool_name: str, params: Dict, context: Dict) -> Dict:
    """Inject context data into tool parameters."""
    
    if tool_name in ["summarizer", "sentiment", "trends"]:
        if "news" in context:
            params["news_items"] = context["news"]
    
    # MAS tools context injection - match actual function signatures
    if tool_name == "web_scraper":
        # scrape_url() expects a single 'url' string
        if "news" in context and "url" not in params:
            # Prioritize direct article URLs over Google News redirects
            direct_urls = [item.get("link") for item in context["news"] 
                          if item.get("link") and not item["link"].startswith("https://news.google.com/")]
            all_urls = [item.get("link") for item in context["news"] if item.get("link")]
            urls = direct_urls or all_urls
            if urls:
                params["url"] = urls[0]  # Single URL for scrape_url()
        # Remove 'urls' if accidentally set by mock planner
        params.pop("urls", None)
    
    if tool_name == "entity_extractor":
        # extract_entities_from_articles() expects 'articles' (List[Dict])
        if "news" in context and "articles" not in params:
            params["articles"] = context["news"]
        # Remove 'text' if accidentally set by mock planner
        params.pop("text", None)
    
    if tool_name == "image_analyzer":
        # analyze_article_images() expects 'articles' (List[Dict]) with 'images' field
        if "articles" not in params:
            # First: try to use web_scraper results which include extracted images
            scraped = context.get("scraped_articles") or context.get("web_scraper")
            if scraped and isinstance(scraped, dict):
                # Scraper returns individual article result
                scraped_list = scraped.get("articles", [scraped]) if isinstance(scraped.get("articles"), list) else [scraped]
                params["articles"] = [a for a in scraped_list if a.get("images")]
            
            # Fallback: pass article links so analyzer can extract og:image from pages
            if not params.get("articles") and "news" in context:
                # Prioritize direct article links (gnews, tavily) over Google News redirects
                direct_articles = [item for item in context["news"] if not item.get("link", "").startswith("https://news.google.com/")]
                if not direct_articles:
                    direct_articles = context["news"]
                enriched = []
                for item in direct_articles[:5]:
                    link = item.get("link", "")
                    if link:
                        # Carry over any existing image URL (e.g. from GNews)
                        img_list = []
                        if item.get("image"):
                            img_list.append(item["image"])
                        enriched.append({
                            "title": item.get("title", ""),
                            "link": link,
                            "image": item.get("image", ""),
                            "images": img_list
                        })
                params["articles"] = enriched
        # Remove wrong param names
        params.pop("image_urls", None)
    
    if tool_name == "research_assistant":
        # comprehensive_research() expects 'topic' (str)
        if "query" in params and "topic" not in params:
            params["topic"] = params.pop("query")
        if "topic" not in params:
            # Try to derive topic from user input
            params["topic"] = context.get("user_input", "AI news")
    
    if tool_name == "social_monitor":
        # monitor_social_media() expects 'topic' (str)
        if "topic" not in params:
            params["topic"] = context.get("user_input", "AI")
    
    if tool_name == "exporter":
        params["data"] = {
            "news": context.get("news", []),
            "summary": context.get("summary"),
            "sentiment": context.get("sentiment"),
            "trends": context.get("trends"),
            "entities": context.get("entities"),
            "images": context.get("images"),
            "social": context.get("social"),
            "research": context.get("research")
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
    # MAS tools
    elif tool_name == "web_scraper":
        context["scraped_articles"] = output
    elif tool_name == "entity_extractor":
        context["entities"] = output
    elif tool_name == "image_analyzer":
        context["images"] = output
    elif tool_name == "social_monitor":
        context["social"] = output
    elif tool_name == "research_assistant":
        context["research"] = output
    elif tool_name == "exporter":
        context["exported_file"] = output

"""FastAPI routes for Nova Intelligence Agent.

v3: Routes through LangGraph pipeline with v2 fallback.
Phase 3: Memory endpoints + Continuous monitoring.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import httpx
import os
import io
import traceback
from dotenv import load_dotenv

from app.models.schemas import CommandRequest, TaskPlan
from app.agents.planner_agent import plan_task
from app.agents.executor_agent import execute_plan
from app.memory.store import save_plan, get_recent_plans, get_recent_results
from app.core.tool_registry import list_tools
from app.tools.exporter import export_data

# v3 Phase 3: Memory manager and WebSocket
try:
    from app.memory.manager import memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from app.api.ws import continuous_engine
    CONTINUOUS_AVAILABLE = True
except ImportError:
    CONTINUOUS_AVAILABLE = False

# v3: LangGraph pipeline (graceful import — falls back to v2 if not available)
try:
    from app.graph.nova_graph import run_graph
    NOVA_V3_AVAILABLE = True
except ImportError:
    NOVA_V3_AVAILABLE = False
    print("[ROUTES] LangGraph not available — using v2 executor")

load_dotenv()

# Configuration: set USE_V3_GRAPH=true to enable LangGraph pipeline
USE_V3_GRAPH = os.getenv("USE_V3_GRAPH", "true").lower() == "true"

router = APIRouter()


@router.post("/command")
async def process_command(request: CommandRequest) -> Dict[str, Any]:
    """
    Main endpoint: Process a voice/text command.
    
    v3 Flow: Text → LangGraph (supervisor → pipelines → fusion → critic → memory → output)
    v2 Fallback: Text → Planner → Task JSON → Executor → Results
    
    Never returns HTTP 500 - always returns a response with error details.
    """
    # ═══ v3 PATH: LangGraph Pipeline ═══
    if USE_V3_GRAPH and NOVA_V3_AVAILABLE:
        try:
            print(f"[API v3] Running graph for: {request.text}")

            # Extract feature toggles from request
            toggles = request.feature_toggles or {}

            result = await run_graph(
                query=request.text,
                feature_toggles=toggles
            )

            # run_graph returns the final_report which is v2-compatible
            plan_dict = result.get("plan", {
                "intent": result.get("intent", ""),
                "domain": result.get("domain", ""),
            })

            print(f"[API v3] Graph complete. Success: {result.get('success', False)}")

            return {
                "success": result.get("success", False),
                "plan": plan_dict,
                "result": result,
                "errors": result.get("errors", []),
                "v3": True,  # Signal to frontend that v3 was used
            }

        except Exception as e:
            print(f"[API v3 ERROR] Graph failed, falling back to v2: {e}")
            # Fall through to v2 path

    # ═══ v2 PATH: Legacy Planner → Executor ═══
    plan_dict = None
    result = None
    errors = []
    
    # Step 1: Plan the task
    try:
        print(f"[API v2] Planning task for: {request.text}")
        plan_dict = plan_task(request.text)
        save_plan(plan_dict, request.text)
        print(f"[API v2] Plan created: {plan_dict.get('intent', 'unknown')}")
    except Exception as e:
        error_msg = f"Planning failed: {str(e)}"
        print(f"[API ERROR] {error_msg}\n{traceback.format_exc()}")
        errors.append(error_msg)
        plan_dict = {
            "intent": f"Get {request.text} news",
            "domain": request.text.split()[0] if request.text else "ai",
            "steps": [
                {"tool": "news_fetcher", "params": {"topic": request.text or "ai", "sources": ["google"], "limit": 5}},
                {"tool": "exporter", "params": {"filename": "report", "format": "json"}}
            ]
        }
    
    # Step 2: Execute the plan
    try:
        plan = TaskPlan(**plan_dict)
        print(f"[API v2] Executing {len(plan.steps)} steps...")
        result = execute_plan(plan)
        print(f"[API v2] Execution complete. Success: {result.get('success', False)}")
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        print(f"[API ERROR] {error_msg}\n{traceback.format_exc()}")
        errors.append(error_msg)
        result = {
            "intent": plan_dict.get("intent", "unknown"),
            "domain": plan_dict.get("domain", "unknown"),
            "tools_executed": [],
            "data": {},
            "errors": errors,
            "skipped": [],
            "fallbacks_used": [],
            "regenerated": [],
            "success": False
        }
    
    return {
        "success": len(errors) == 0 and result.get("success", False),
        "plan": plan_dict,
        "result": result,
        "errors": errors,
        "v3": False,
    }


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Get agent capabilities for UI display."""
    return {
        "name": "Nova Intelligence Agent",
        "version": "3.0",
        "engine": "LangGraph" if (USE_V3_GRAPH and NOVA_V3_AVAILABLE) else "v2-executor",
        "features": [
            {"id": "multi_source", "name": "Multi-Source News", "icon": "📰"},
            {"id": "ai_summary", "name": "AI Summary", "icon": "🧠"},
            {"id": "sentiment", "name": "Sentiment Analysis", "icon": "💭"},
            {"id": "trends", "name": "Trend Detection", "icon": "📊"},
            {"id": "web_scraper", "name": "Web Scraper", "icon": "🌐"},
            {"id": "entity_extractor", "name": "Entity Network", "icon": "👤"},
            {"id": "image_analyzer", "name": "Image Intelligence", "icon": "🖼️"},
            {"id": "social_monitor", "name": "Social Monitor", "icon": "📱"},
            {"id": "research_assistant", "name": "Research Assistant", "icon": "📚"},
            {"id": "export", "name": "Package Builder", "icon": "📦"},
        ],
        "v3_capabilities": {
            "multi_pipeline": True,
            "critic_agent": True,
            "memory_system": True,
            "supervisor_agent": True,
            "fusion_layer": True,
        },
        "sources": ["Tavily", "GNews", "Google News RSS", "Reddit", "arXiv", "GitHub", "StackOverflow"],
        "domains": ["AI", "Tech", "Crypto", "Research", "Business", "Politics", "Science"],
        "export_formats": ["JSON", "Markdown", "CSV", "Word", "PDF"],
        "tools": list_tools()
    }


@router.get("/history")
async def get_history() -> Dict[str, Any]:
    """Get recent command history."""
    return {
        "plans": get_recent_plans(10),
        "results": get_recent_results(10)
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "agent": "Nova Intelligence Agent"}


# ============ EXPORT API ============

@router.post("/export")
async def export_report(request: Dict[str, Any]):
    """
    Export filtered report data in specified format.
    
    Body: {"data": {...}, "format": "json|markdown|csv|docx|pdf", "filename": "report"}
    Returns: Downloadable file
    """
    data = request.get("data", {})
    format_type = request.get("format", "json")
    filename = request.get("filename", "nova_report")
    
    if not data:
        raise HTTPException(status_code=400, detail="No data to export")
    
    try:
        # Generate the export file
        filepath = export_data(data, filename, format_type)
        
        # Determine content type
        content_types = {
            "json": "application/json",
            "markdown": "text/markdown",
            "csv": "text/csv",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pdf": "application/pdf"
        }
        content_type = content_types.get(format_type, "application/octet-stream")
        
        # Read file and return as streaming response
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Determine file extension
        extensions = {"json": "json", "markdown": "md", "csv": "csv", "docx": "docx", "pdf": "pdf"}
        ext = extensions.get(format_type, "txt")
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.{ext}"',
                "Content-Length": str(len(content))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ============ TRANSLATION API (MyMemory - Free) ============

@router.post("/translate")
async def translate_text(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate text using MyMemory API (free, no key required).
    
    Body: {"text": "Hello", "from": "en", "to": "hi"}
    """
    text = request.get("text", "")
    source_lang = request.get("from", "en")
    target_lang = request.get("to", "hi")
    
    if not text:
        return {"success": False, "error": "No text provided"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.mymemory.translated.net/get",
                params={
                    "q": text[:500],  # Limit to 500 chars per request
                    "langpair": f"{source_lang}|{target_lang}"
                }
            )
            data = response.json()
            
            if data.get("responseStatus") == 200:
                translated = data.get("responseData", {}).get("translatedText", text)
                return {
                    "success": True,
                    "original": text,
                    "translated": translated,
                    "from": source_lang,
                    "to": target_lang
                }
            else:
                return {"success": False, "error": "Translation failed"}
                
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/languages")
async def get_languages() -> Dict[str, Any]:
    """Get available translation languages."""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "it", "name": "Italian"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "bn", "name": "Bengali"},
            {"code": "mr", "name": "Marathi"},
            {"code": "gu", "name": "Gujarati"},
            {"code": "pa", "name": "Punjabi"},
        ]
    }


# ============ DICTIONARY API (Merriam-Webster) ============

@router.get("/dictionary/{word}")
async def get_definition(word: str) -> Dict[str, Any]:
    """
    Get word definition from Merriam-Webster API.
    """
    api_key = os.getenv("MERRIAM_WEBSTER_API_KEY", "")
    
    if not api_key:
        # Fallback to Free Dictionary API if no key
        return await _get_free_dictionary(word)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}",
                params={"key": api_key}
            )
            data = response.json()
            
            if not data or isinstance(data[0], str):
                # Suggestions returned instead of definitions
                return {
                    "success": False,
                    "word": word,
                    "suggestions": data[:5] if isinstance(data, list) else [],
                    "error": "Word not found"
                }
            
            # Parse Merriam-Webster response
            entry = data[0]
            definitions = []
            
            if "shortdef" in entry:
                definitions = entry["shortdef"]
            elif "def" in entry:
                for sense in entry.get("def", []):
                    for sseq in sense.get("sseq", []):
                        for item in sseq:
                            if item[0] == "sense":
                                dt = item[1].get("dt", [])
                                for d in dt:
                                    if d[0] == "text":
                                        definitions.append(d[1].replace("{bc}", "").strip())
            
            return {
                "success": True,
                "word": word,
                "partOfSpeech": entry.get("fl", ""),
                "definitions": definitions[:3],
                "source": "Merriam-Webster"
            }
            
    except Exception as e:
        return await _get_free_dictionary(word)


async def _get_free_dictionary(word: str) -> Dict[str, Any]:
    """Fallback to Free Dictionary API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            )
            
            if response.status_code != 200:
                return {"success": False, "word": word, "error": "Word not found"}
            
            data = response.json()
            entry = data[0]
            
            definitions = []
            part_of_speech = ""
            
            for meaning in entry.get("meanings", []):
                if not part_of_speech:
                    part_of_speech = meaning.get("partOfSpeech", "")
                for defn in meaning.get("definitions", [])[:2]:
                    definitions.append(defn.get("definition", ""))
            
            return {
                "success": True,
                "word": word,
                "partOfSpeech": part_of_speech,
                "definitions": definitions[:3],
                "source": "Free Dictionary"
            }
            
    except Exception as e:
        return {"success": False, "word": word, "error": str(e)}


# ============ PHASE 3: CONTINUOUS MONITORING ============

class MonitorRequest(BaseModel):
    topic: str
    interval_minutes: int = 30
    duration_hours: int = 24
    depth: str = "standard"


@router.post("/monitor")
async def start_monitor(request: MonitorRequest) -> Dict[str, Any]:
    """Start a continuous intelligence monitor.
    
    Body: {"topic": "Tesla", "interval_minutes": 30, "duration_hours": 24}
    """
    if not CONTINUOUS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Continuous mode not available")

    # Inject graph runner if not set
    if continuous_engine._graph_runner is None and NOVA_V3_AVAILABLE:
        continuous_engine.set_graph_runner(run_graph)

    task_id = await continuous_engine.start_monitor(
        topic=request.topic,
        interval_minutes=request.interval_minutes,
        duration_hours=request.duration_hours,
        depth=request.depth,
    )

    return {
        "success": True,
        "monitor_id": task_id,
        "message": f"Monitoring '{request.topic}' every {request.interval_minutes}min for {request.duration_hours}h",
    }


@router.delete("/monitor/{monitor_id}")
async def stop_monitor(monitor_id: str) -> Dict[str, Any]:
    """Stop a running monitor."""
    if not CONTINUOUS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Continuous mode not available")

    success = continuous_engine.stop_monitor(monitor_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Monitor '{monitor_id}' not found")

    return {"success": True, "message": f"Monitor '{monitor_id}' stopped"}


@router.get("/monitor/active")
async def list_active_monitors() -> Dict[str, Any]:
    """List all active monitoring tasks."""
    if not CONTINUOUS_AVAILABLE:
        return {"monitors": [], "message": "Continuous mode not available"}

    return {
        "monitors": continuous_engine.get_active_monitors(),
        "total": len(continuous_engine.get_active_monitors()),
    }


# ============ PHASE 3: MEMORY ENDPOINTS ============

@router.get("/memory/query")
async def query_memory(topic: str, limit: int = 10) -> Dict[str, Any]:
    """Search past intelligence by topic.
    
    Query: /api/memory/query?topic=Tesla&limit=5
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory system not available")

    history = memory_manager.long_term.get_topic_history(topic, limit=limit)

    # Slim down results (don't send full result_json)
    results = []
    for entry in history:
        results.append({
            "query": entry.get("query", ""),
            "depth": entry.get("depth", "standard"),
            "critic_score": entry.get("critic_score", 0),
            "confidence": entry.get("confidence", 0.0),
            "pipelines": entry.get("pipelines", "[]"),
            "created_at": entry.get("created_at", ""),
            "duration_ms": entry.get("duration_ms", 0),
        })

    return {
        "topic": topic,
        "results": results,
        "count": len(results),
    }


@router.get("/memory/compare")
async def compare_memory(topic: str, days: int = 7) -> Dict[str, Any]:
    """Compare topic intelligence over time.
    
    Query: /api/memory/compare?topic=Tesla&days=7
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory system not available")

    comparison = memory_manager.compare_topic(topic, days=days)
    entity_history = memory_manager.get_entity_history(topic, limit=10)
    trend_changes = memory_manager.detect_trend_changes(topic, hours=days * 24)

    return {
        "topic": topic,
        "comparison": comparison,
        "entity_history": entity_history,
        "trend_changes": trend_changes,
    }


@router.get("/memory/stats")
async def memory_stats() -> Dict[str, Any]:
    """Get memory system statistics."""
    if not MEMORY_AVAILABLE:
        return {"available": False}

    stats = memory_manager.get_stats()
    stats["available"] = True
    return stats


# ============ PHASE 3: GRAPH STATUS ============

@router.get("/graph/status")
async def graph_status() -> Dict[str, Any]:
    """Get v3 graph engine status."""
    from datetime import datetime
    return {
        "status": "ok",
        "time_str": datetime.utcnow().strftime("%H:%M UTC"),
        "v3_available": NOVA_V3_AVAILABLE,
        "v3_enabled": USE_V3_GRAPH,
        "engine": "LangGraph" if (USE_V3_GRAPH and NOVA_V3_AVAILABLE) else "v2-executor",
        "memory_available": MEMORY_AVAILABLE,
        "continuous_available": CONTINUOUS_AVAILABLE,
        "memory_stats": memory_manager.get_stats() if MEMORY_AVAILABLE else None,
        "active_monitors": continuous_engine.get_active_monitors() if CONTINUOUS_AVAILABLE else [],
    }

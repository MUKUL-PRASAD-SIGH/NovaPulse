"""FastAPI routes for Nova Intelligence Agent."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import httpx
import os
from dotenv import load_dotenv

from app.models.schemas import CommandRequest, TaskPlan
from app.agents.planner_agent import plan_task
from app.agents.executor_agent import execute_plan
from app.memory.store import save_plan, get_recent_plans, get_recent_results
from app.core.tool_registry import list_tools

load_dotenv()

router = APIRouter()


@router.post("/command")
async def process_command(request: CommandRequest) -> Dict[str, Any]:
    """
    Main endpoint: Process a voice/text command.
    
    Flow: Text → Planner → Task JSON → Executor → Results
    """
    try:
        # Step 1: Plan the task
        plan_dict = plan_task(request.text)
        save_plan(plan_dict, request.text)
        
        # Step 2: Execute the plan
        plan = TaskPlan(**plan_dict)
        result = execute_plan(plan)
        
        return {
            "success": True,
            "plan": plan_dict,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Get agent capabilities for UI display."""
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
        "sources": ["Google News", "TechCrunch", "The Verge"],
        "domains": ["AI", "Tech", "Crypto", "Research", "Business"],
        "export_formats": ["JSON", "Markdown", "CSV"],
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

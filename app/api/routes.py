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

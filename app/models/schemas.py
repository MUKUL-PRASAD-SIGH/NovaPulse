"""Pydantic schemas for Nova Intelligence Agent."""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ToolStep(BaseModel):
    """Single step in the execution plan."""
    tool: str
    params: Dict[str, Any] = {}


class TaskPlan(BaseModel):
    """Complete task plan from Planner Agent."""
    intent: str
    domain: str = "ai"
    steps: List[ToolStep]


class CommandRequest(BaseModel):
    """API request from frontend.
    
    v3: Added feature_toggles for supervisor routing decisions.
    """
    text: str
    feature_toggles: Optional[Dict[str, bool]] = None


class CommandResponse(BaseModel):
    """API response to frontend."""
    plan: Dict[str, Any]
    result: Dict[str, Any]

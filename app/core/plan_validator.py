"""Plan validator for task JSON."""
from typing import Tuple
from app.models.schemas import TaskPlan

VALID_TOOLS = ["news_fetcher", "summarizer", "sentiment", "trends", "exporter"]

def validate_plan(plan_dict: dict) -> Tuple[bool, str]:
    try:
        if "steps" not in plan_dict or not plan_dict["steps"]:
            return False, "No steps defined"
        for step in plan_dict["steps"]:
            if step.get("tool") not in VALID_TOOLS:
                return False, f"Unknown tool: {step.get('tool')}"
        TaskPlan(**plan_dict)
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def get_valid_tools() -> list:
    return VALID_TOOLS.copy()

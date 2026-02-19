"""Memory store for plans, results, and logs."""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

MEMORY_DIR = os.path.join(os.path.dirname(__file__))

def _get_path(filename: str) -> str:
    return os.path.join(MEMORY_DIR, filename)

def _load(filename: str) -> List[Dict]:
    path = _get_path(filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def _save(filename: str, data: List[Dict], max_entries: int = 100):
    path = _get_path(filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data[-max_entries:], f, indent=2)

def save_plan(plan: Dict, user_input: str):
    plans = _load("plans.json")
    plans.append({"timestamp": datetime.now().isoformat(), "user_input": user_input, "plan": plan})
    _save("plans.json", plans, 50)

def save_result(result: Dict):
    results = _load("results.json")
    results.append({"timestamp": datetime.now().isoformat(), "result": result})
    _save("results.json", results, 50)

def log(level: str, message: str, data: Optional[Dict] = None):
    logs = _load("logs.json")
    entry = {"timestamp": datetime.now().isoformat(), "level": level.upper(), "message": message}
    if data:
        entry["data"] = data
    logs.append(entry)
    _save("logs.json", logs, 200)

def get_recent_plans(limit: int = 10) -> List[Dict]:
    return _load("plans.json")[-limit:]

def get_recent_results(limit: int = 10) -> List[Dict]:
    return _load("results.json")[-limit:]

async def log_tool_execution(tool_name: str = "", success: bool = True, duration: float = 0.0, 
                              metadata: Optional[Dict] = None, **kwargs):
    """Log tool execution for monitoring and debugging.
    
    Accepts both old-style (tool_name, success, duration) and 
    new-style (tool_name, params, result) keyword arguments.
    """
    log_data = {
        "tool": tool_name,
        "success": success,
        "duration_seconds": duration,
    }
    if metadata:
        log_data.update(metadata)
    # Support new-style kwargs from MAS tools (params=..., result=...)
    if "params" in kwargs:
        log_data["params"] = kwargs["params"]
    if "result" in kwargs:
        log_data["result"] = kwargs["result"]
    
    log("INFO" if success else "ERROR", f"Tool {tool_name} executed", log_data)


"""Memory store for plans, results, and logs."""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.core.runtime_paths import get_runtime_directory


MEMORY_DIR = get_runtime_directory(
    "memory",
    env_var="NOVA_DATA_DIR",
    local_directory=Path(__file__).parent,
)
_STORE_LOCK = threading.RLock()

def _get_path(filename: str) -> str:
    return str(MEMORY_DIR / filename)

def _load(filename: str) -> List[Dict]:
    path = _get_path(filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, TypeError):
            return []
    return []

def _save(filename: str, data: List[Dict], max_entries: int = 100) -> bool:
    path = _get_path(filename)
    temp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with _STORE_LOCK:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data[-max_entries:], f, indent=2, default=str)
            os.replace(temp_path, path)
        return True
    except OSError:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return False

def save_plan(plan: Dict, user_input: str) -> bool:
    with _STORE_LOCK:
        plans = _load("plans.json")
        plans.append({"timestamp": datetime.now().isoformat(), "user_input": user_input, "plan": plan})
        return _save("plans.json", plans, 50)

def save_result(result: Dict) -> bool:
    with _STORE_LOCK:
        results = _load("results.json")
        results.append({"timestamp": datetime.now().isoformat(), "result": result})
        return _save("results.json", results, 50)

def log(level: str, message: str, data: Optional[Dict] = None) -> bool:
    """Best-effort runtime logging that can never stop the pipeline."""
    with _STORE_LOCK:
        logs = _load("logs.json")
        entry = {"timestamp": datetime.now().isoformat(), "level": level.upper(), "message": message}
        if data:
            entry["data"] = data
        logs.append(entry)
        return _save("logs.json", logs, 200)

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


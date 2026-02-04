"""Executor Agent for Nova Intelligence Agent.

Executes task plans by running tools in sequence.
Handles context passing between tools.
"""
from typing import Dict, Any, List
from app.core.tool_registry import get_tool, list_tools
from app.models.schemas import TaskPlan, ToolStep
from app.memory.store import save_result, log


def execute_plan(plan: TaskPlan) -> Dict[str, Any]:
    """
    Execute a task plan by running each tool step.
    
    Args:
        plan: TaskPlan object with intent and steps
    
    Returns:
        Execution result with data, tools run, and any errors
    """
    log("INFO", f"Executing plan: {plan.intent}", {"steps": len(plan.steps)})
    
    result = {
        "intent": plan.intent,
        "domain": plan.domain,
        "tools_executed": [],
        "data": {},
        "errors": [],
        "success": True
    }
    
    # Context for passing data between tools
    context: Dict[str, Any] = {}
    
    for i, step in enumerate(plan.steps):
        tool_result = _execute_step(step, context, i)
        
        if tool_result["success"]:
            result["tools_executed"].append({
                "tool": step.tool,
                "success": True
            })
            # Update context with tool output
            _update_context(context, step.tool, tool_result["output"])
        else:
            result["errors"].append(tool_result["error"])
            result["tools_executed"].append({
                "tool": step.tool,
                "success": False,
                "error": tool_result["error"]
            })
    
    # Set final data from context
    result["data"] = context
    result["success"] = len(result["errors"]) == 0
    
    # Save to memory
    save_result(result)
    log("INFO", f"Execution complete: {len(result['tools_executed'])} tools run")
    
    return result


def _execute_step(step: ToolStep, context: Dict, step_index: int) -> Dict:
    """Execute a single tool step."""
    tool_fn = get_tool(step.tool)
    
    if not tool_fn:
        return {
            "success": False,
            "error": f"Tool not found: {step.tool}",
            "output": None
        }
    
    try:
        # Prepare parameters with context injection
        params = dict(step.params)
        params = _inject_context(step.tool, params, context)
        
        log("DEBUG", f"Running tool: {step.tool}", {"params": list(params.keys())})
        
        # Execute the tool
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
    
    # Tools that need news_items from context - ALWAYS override
    if tool_name in ["summarizer", "sentiment", "trends"]:
        if "news" in context:
            # Always use actual news data from context (Nova puts empty/placeholder values)
            params["news_items"] = context["news"]
    
    # Exporter needs all collected data - always override with context
    if tool_name == "exporter":
        # Always build data from context (Nova might provide empty/invalid data)
        params["data"] = {
            "news": context.get("news", []),
            "summary": context.get("summary"),
            "sentiment": context.get("sentiment"),
            "trends": context.get("trends")
        }
        # Clean None values
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

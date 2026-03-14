"""Nova v3 — Edge Routing Logic.

Defines conditional edge functions used by the LangGraph StateGraph.
These functions determine which path the execution takes at branching points.
"""

from typing import Dict, Any, List


def route_after_supervisor(state: Dict[str, Any]) -> str:
    """Route from supervisor to the appropriate pipeline fan-out.
    
    Returns a key that maps to the next node(s) in the graph.
    The graph definition maps these keys to actual node names.
    """
    # If somehow we have no query, jump straight to output
    if not state.get("query"):
        return "output_only"

    return "run_pipelines"


def should_retry(state: Dict[str, Any]) -> str:
    """Decide whether critic wants a retry or approves the output.
    
    Returns:
        "retry"  - quality too low, try again (goes back to fusion)
        "accept" - quality acceptable or max retries reached (goes to memory)
    """
    if state.get("critic_approved", True):
        return "accept"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count >= max_retries:
        return "accept"  # Accept anyway after exhausting retries

    return "retry"

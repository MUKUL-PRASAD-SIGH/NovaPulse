"""Nova v3 — Main LangGraph Definition.

Builds the complete intelligence graph:

    supervisor → pipelines (parallel) → fusion → critic → memory → output
                                                   ↑         |
                                                   └─ retry ─┘

Usage:
    from app.graph.nova_graph import nova_graph
    
    result = await nova_graph.ainvoke({
        "query": "Tesla news with sentiment",
        "feature_toggles": {"summarizer": True, "sentiment": True}
    })
    
    final_report = result["final_report"]
"""

import asyncio
from typing import Dict, Any

from app.graph.state import NovaState
from app.graph.nodes.supervisor_node import supervisor_node
from app.graph.nodes.news_pipeline_node import news_pipeline_node
from app.graph.nodes.social_pipeline_node import social_pipeline_node
from app.graph.nodes.research_pipeline_node import research_pipeline_node
from app.graph.nodes.market_pipeline_node import market_pipeline_node
from app.graph.nodes.fusion_node import fusion_node
from app.graph.nodes.critic_node import critic_node
from app.graph.nodes.memory_node import memory_node
from app.graph.nodes.output_node import output_node
from app.graph.edges.routing import route_after_supervisor, should_retry

from app.memory.store import log


# ═══════════════════════════════════════════════════════
#  PIPELINE FAN-OUT NODE
#  Runs all active pipelines in parallel via asyncio
# ═══════════════════════════════════════════════════════

async def pipeline_fanout_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run all activated pipelines in parallel.
    
    This is a single LangGraph node that internally fans out to
    multiple async pipeline functions. This avoids the complexity
    of LangGraph's native parallelism for Phase 1 while still
    getting concurrent execution.
    
    Reads:  active_pipelines (set by supervisor)
    Writes: news_result, social_result, research_result, market_result, execution_trace
    """
    active = state.get("active_pipelines", ["news"])

    # Build task list
    tasks = {}
    
    # News always runs
    tasks["news"] = news_pipeline_node(state)

    # Conditional pipelines
    if "social" in active:
        tasks["social"] = social_pipeline_node(state)
    if "research" in active:
        tasks["research"] = research_pipeline_node(state)
    if "market" in active:
        tasks["market"] = market_pipeline_node(state)

    # Execute all in parallel
    keys = list(tasks.keys())
    results = await asyncio.gather(
        *tasks.values(),
        return_exceptions=True
    )

    # Merge all pipeline results into state
    merged = {}
    all_traces = list(state.get("execution_trace", []))

    for key, result in zip(keys, results):
        if isinstance(result, Exception):
            log("ERROR", f"Pipeline {key} crashed: {result}")
            merged[f"{key}_result"] = {
                "success": False,
                "error": str(result),
            }
            all_traces.append({
                "node": f"{key}_pipeline",
                "status": "crashed",
                "error": str(result),
            })
        elif isinstance(result, dict):
            # Extract pipeline-specific result and trace entries
            for k, v in result.items():
                if k == "execution_trace":
                    # Flatten trace entries from pipeline
                    all_traces.extend(v[len(state.get("execution_trace", [])):])
                else:
                    merged[k] = v

    merged["execution_trace"] = all_traces
    return merged


# ═══════════════════════════════════════════════════════
#  GRAPH BUILDER
# ═══════════════════════════════════════════════════════

def build_nova_graph():
    """Build and compile the Nova v3 intelligence graph.
    
    Graph structure:
        supervisor → pipeline_fanout → fusion → critic → [retry|accept]
                                                              ↓
                                          retry → fusion (loop back)
                                          accept → memory → output → END    
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "LangGraph is required for Nova v3. Install with: "
            "pip install langgraph langchain-core"
        )

    graph = StateGraph(NovaState)

    # ── ADD NODES ──
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("pipeline_fanout", pipeline_fanout_node)
    graph.add_node("fusion", fusion_node)
    graph.add_node("critic", critic_node)
    graph.add_node("memory", memory_node)
    graph.add_node("output", output_node)

    # ── ENTRY POINT ──
    graph.set_entry_point("supervisor")

    # ── EDGES ──

    # Supervisor → Pipeline fan-out (always)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "run_pipelines": "pipeline_fanout",
            "output_only": "output",
        }
    )

    # Pipeline fan-out → Fusion
    graph.add_edge("pipeline_fanout", "fusion")

    # Fusion → Critic
    graph.add_edge("fusion", "critic")

    # Critic → conditional: retry or accept
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "fusion",      # Re-run fusion with critic feedback
            "accept": "memory",     # Approved — save to memory
        }
    )

    # Memory → Output → END
    graph.add_edge("memory", "output")
    graph.add_edge("output", END)

    # ── COMPILE ──
    return graph.compile()


# ═══════════════════════════════════════════════════════
#  SINGLETON GRAPH INSTANCE
# ═══════════════════════════════════════════════════════

# Lazy initialization to avoid import errors if langgraph isn't installed
_nova_graph = None


def get_nova_graph():
    """Get or build the Nova graph (singleton)."""
    global _nova_graph
    if _nova_graph is None:
        _nova_graph = build_nova_graph()
        log("INFO", "Nova v3 graph compiled successfully")
    return _nova_graph


async def run_graph(query: str, feature_toggles: Dict[str, bool] = None) -> Dict[str, Any]:
    """High-level entry point: run a query through the Nova v3 graph.
    
    Args:
        query: User's input text
        feature_toggles: Dict of {tool_name: enabled} from UI
        
    Returns:
        The final_report dict (v2-compatible + v3 metadata)
    """
    graph = get_nova_graph()

    initial_state = {
        "query": query,
        "feature_toggles": feature_toggles or {},
        "execution_trace": [],
        "retry_count": 0,
    }

    try:
        result = await graph.ainvoke(initial_state)
        return result.get("final_report", result)
    except Exception as e:
        log("ERROR", f"Graph execution failed: {e}")
        return {
            "intent": f"Analyze: {query}",
            "domain": "unknown",
            "tools_executed": [],
            "data": {},
            "errors": [f"Graph execution failed: {str(e)}"],
            "success": False,
            "v3_meta": {
                "depth": "error",
                "confidence": 0.0,
                "quality_badge": "raw",
                "graph_trace": [],
            }
        }

"""Nova v3 — Global State Schema.

NovaState is the single source of truth flowing through the entire LangGraph.
Every node reads from and writes to this state.

Design Principles:
- All fields are Optional (total=False) — nodes only set what they produce
- Pipelines write to their own namespace (news_result, social_result, etc.)
- Critic writes approval status; edges read it for routing
- execution_trace is append-only — every node records its timing
"""

from typing import TypedDict, Optional, List, Dict, Any
from enum import Enum


class AnalysisDepth(str, Enum):
    """How deep the system should analyze."""
    QUICK = "quick"         # News pipeline only, no critic
    STANDARD = "standard"   # News + toggled features, no critic
    DEEP = "deep"           # All pipelines + critic + memory compare


class NovaState(TypedDict, total=False):
    """Global state shared across all LangGraph nodes.
    
    This replaces the implicit 'context' dict from executor_agent.py
    with a well-typed, documented schema.
    """

    # ══════════════════════════════════════════
    #  INPUT (set by API route before graph invocation)
    # ══════════════════════════════════════════
    query: str                              # Original user query text
    feature_toggles: Dict[str, bool]        # UI toggle states {tool_name: enabled}
    
    # ══════════════════════════════════════════
    #  SUPERVISOR DECISIONS (set by supervisor_node)
    # ══════════════════════════════════════════
    depth: str                              # AnalysisDepth value as string
    active_pipelines: List[str]             # Which pipelines to run: ["news", "social", ...]
    enable_critic: bool                     # Whether to run quality gate
    max_retries: int                        # Max retry attempts if critic fails
    plan: Dict[str, Any]                    # Execution plan (from planner_agent)

    # ══════════════════════════════════════════
    #  PIPELINE OUTPUTS (set by pipeline nodes)
    # ══════════════════════════════════════════
    news_result: Optional[Dict[str, Any]]       # Pipeline 1: news intelligence
    social_result: Optional[Dict[str, Any]]     # Pipeline 2: social intelligence
    research_result: Optional[Dict[str, Any]]   # Pipeline 3: research intelligence
    market_result: Optional[Dict[str, Any]]     # Pipeline 4: market intelligence

    # ══════════════════════════════════════════
    #  FUSION (set by fusion_node)
    # ══════════════════════════════════════════
    fused_report: Optional[Dict[str, Any]]  # Cross-pipeline synthesized report
    narrative: Optional[str]                # Unified narrative text from Nova LLM

    # ══════════════════════════════════════════
    #  CRITIC (set by critic_node)
    # ══════════════════════════════════════════
    critic_score: int                       # 0-100 quality score
    critic_feedback: List[str]              # Improvement suggestions
    critic_approved: bool                   # Pass/fail gate
    retry_count: int                        # Current retry number

    # ══════════════════════════════════════════
    #  MEMORY (set by memory_node)
    # ══════════════════════════════════════════
    memory_context: Optional[Dict[str, Any]]    # Retrieved past data for context
    memory_comparison: Optional[Dict[str, Any]] # Delta vs past results

    # ══════════════════════════════════════════
    #  OUTPUT (set by output_node)
    # ══════════════════════════════════════════
    final_report: Optional[Dict[str, Any]]  # Complete packaged intelligence
    execution_trace: List[Dict[str, Any]]   # Per-node timing & status records
    confidence_score: float                 # Overall confidence 0.0 - 1.0
    error: Optional[str]                    # Top-level error if something went wrong

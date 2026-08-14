"""
LangGraph StateGraph definition. 

Node execution order:
  guardrail -> [load_memory] -> planner -> [guideline_rag || pubmed] ->
  evidence_ranking -> clinical_reasoning -> reflection ->
  clinical_summary -> save_memory -> END

Parallel fan-out uses LangGraph's native parallel node execution.
"""

from __future__ import annotations

import logging
from functools import lru_cache 

from langgraph.graph import END, StateGraph

from app.agents.clinical_reasoning_agent import run_clinical_reasoning
from app.agents.clinical_summary_agent import run_clinical_summary
from app.agents.guardrail_agent import run_guardrail
from app.agents.planner_agent import run_planner
from app.agents.reflection_agent import run_reflection
from app.nodes import (
    load_memory_node,
    run_evidence_ranking,
    run_guideline_rag,
    run_pubmed,
    save_memory_node,
)
from app.graph.edges import (
    route_after_guardrail,
    route_after_planner,
    route_after_reflection,
)
from app.graph.state import ClinicalState

logger = logging.getLogger(__name__)


def _build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""

    graph = StateGraph(ClinicalState)

    # Adding all nodes
    graph.add_node("guardrail", run_guardrail)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("planner", run_planner)

    graph.add_node("guideline_rag", run_guideline_rag)
    graph.add_node("pubmed", run_pubmed)

    graph.add_node("evidence_ranking", run_evidence_ranking)
    graph.add_node("clinical_reasoning", run_clinical_reasoning)
    graph.add_node("reflection", run_reflection)
    graph.add_node("clinical_summary", run_clinical_summary)
    graph.add_node("save_memory", save_memory_node)

    # Entry point
    graph.set_entry_point("guardrail")

    # Adding edges between nodes

    # Guardrail - conditional branch
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail, # function that decides next node
        {
            "load_memory": "load_memory",
            "__end__": END,
        },
    )

    # Memory load to Planner
    graph.add_edge("load_memory", "planner")

    # Planner to conditional retrieval agents
    # Retrieval agents are selected based on the Planner's routing decision.
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "guideline_rag": "guideline_rag",
            "pubmed": "pubmed",
        },
    )

    # Retrieval fan-in to Evidence Ranking
    graph.add_edge("guideline_rag", "evidence_ranking")
    graph.add_edge("pubmed", "evidence_ranking")

    # Evidence Ranking to Clinical Reasoning
    graph.add_edge("evidence_ranking", "clinical_reasoning")

    # Clinical Reasoning to Reflection
    graph.add_edge("clinical_reasoning", "reflection")

    # Reflection to conditional branch
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "planner": "planner",                  # retry loop via planner
            "clinical_summary": "clinical_summary", # proceed
        },
    )

    # Clinical Summary - Memory Save - END 
    graph.add_edge("clinical_summary", "save_memory")
    graph.add_edge("save_memory", END)

    logger.info("LangGraph StateGraph compiled successfully.")
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled LangGraph graph (cached singleton)."""
    return _build_graph()

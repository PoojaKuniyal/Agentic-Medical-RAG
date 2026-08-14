"""
Conditional edge functions for the LangGraph StateGraph.

These functions determine which node to visit next based on current state.

Edges defined here:
  1. after_guardrail     — route to END (rejected) or load_memory (safe)
  2. after_planner       — fan-out to retrieval agents based on plan
  3. after_reflection    — route back to retrieval or forward to summary
"""

from __future__ import annotations

import logging
from typing import Literal

from app.graph.state import ClinicalState

logger = logging.getLogger(__name__)


def route_after_guardrail(
    state: ClinicalState,
) -> Literal["load_memory", "__end__"]:
    """
    If the query is safe -> proceed to memory loading.
    If blocked -> end the graph immediately.
    """
    if state.get("is_safe", False):
        logger.debug("[Edge] GuardrailAgent -> load_memory")
        return "load_memory"
    logger.debug("[Edge] GuardrailAgent -> END (blocked)")
    return "__end__"


def route_after_planner(state: ClinicalState) -> list[str]:
    """
    Fan-out: determine which retrieval nodes to invoke in parallel.

    The Planner's plan.retrieval_agents field controls this.
    Both agents run in parallel via LangGraph's parallel Send mechanism.

    Returns a list of node names (used with LangGraph's conditional Send).
    """
    plan = state.get("plan")

    if plan and hasattr(plan, "retrieval_agents"):
        agents = plan.retrieval_agents
    else:
        agents = ["guideline_rag", "pubmed"]

    logger.debug("[Edge] PlannerAgent -> parallel: %s", agents)
    return agents


def route_after_reflection(
    state: ClinicalState,
) -> Literal["planner", "clinical_summary"]:
    """
    If reflection agent requests a retry -> loop back to planner.
    Otherwise -> proceed to clinical summary.
    """
    if state.get("reflection_needed", False):
        logger.debug("[Edge] ReflectionAgent -> planner (retry retrieval)")
        return "planner"
    logger.debug("[Edge] ReflectionAgent -> clinical_summary")
    return "clinical_summary"


"""
Planner Agent — intent classification and routing.

Decides whether to use only GuidelineRAG, only PubMed, or both.
If using GuidelineRAG, decides which Chroma collections to query.

NEVER accesses Chroma or PubMed directly.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ClinicalState, PlannerOutput
from app.llm.factory import get_llm
from app.prompts import PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Fallback plan if LLM output cannot be parsed
_FALLBACK_PLAN = PlannerOutput(
    intent="general_clinical_query",
    clinical_topic="general_clinical",
    retrieval_agents=["guideline_rag", "pubmed"],
    chroma_collections=["clinical_guidelines", "consensus_reports"],
    reasoning="Fallback: query both retrieval agents with all collections.",
)


def run_planner(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Planner Agent.

    Reads:  query, memory_context
    Writes: plan
    """
    query = state["query"]
    memory_context = state.get("memory_context", {})
    logger.info("[PlannerAgent] Classifying intent and building routing plan …")

    llm = get_llm(temperature=0.0)

    history_str = ""
    if memory_context.get("conversation_history"):
        turns = memory_context["conversation_history"][-3:]  # last 3 turns
        history_str = "\n".join(
            f"Q: {t['query']}\nA: {t['summary'][:200]}…" for t in turns
        )

    user_message = (
        f"Clinical query:\n{query}"
        + (f"\n\nRecent session context:\n{history_str}" if history_str else "")
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        plan = PlannerOutput(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("[PlannerAgent] Could not parse plan — using fallback. Error: %s", exc)
        plan = _FALLBACK_PLAN

    logger.info(
        "[PlannerAgent] Plan: agents=%s collections=%s intent='%s'",
        plan.retrieval_agents,
        plan.chroma_collections,
        plan.intent,
    )
    return {
        "plan": plan,
        "planner_intent": plan.intent,
        "retrieval_agents_selected": plan.retrieval_agents,
        "chroma_collections_selected": plan.chroma_collections,
    }

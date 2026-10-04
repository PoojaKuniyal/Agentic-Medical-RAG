"""
Planner Agent — intent classification and routing.

Decides whether to use only GuidelineRAG, only PubMed, or both.
If using GuidelineRAG, decides which Chroma collections to query.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.schemas import PlannerOutput
from app.graph.state import ClinicalState
from app.llm.factory import get_llm
from app.prompts import PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Fallback plan if LLM output cannot be parsed
_FALLBACK_PLAN = PlannerOutput(
    intent="general_clinical_query",
    clinical_topic="general_clinical",
    retrieval_agents=["guideline_rag", "pubmed"],
    chroma_collections=["clinical_guidelines", "consensus_reports", "research_articles"],
    reasoning="Fallback: query both retrieval agents with all collections.",
)


def run_planner(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Planner Agent.
    """
    query = state["query"]
    memory_context = state.get("memory_context", {})
    logger.info("[PlannerAgent] Classifying intent and building routing plan …")

    llm = get_llm(model_tier="fast", temperature=0.0)
    structured_llm = llm.with_structured_output(PlannerOutput)

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

    try:
        plan = structured_llm.invoke(messages)
        if isinstance(plan,dict):
            plan = PlannerOutput(**plan)
        elif not isinstance(plan, PlannerOutput):
            raise ValueError(f'Unexpected plan output type: {type(plan)}')
    except Exception as exc:
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

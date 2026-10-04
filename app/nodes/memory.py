"""
Memory Node — Redis-backed session context.

Two node functions:
  - load_memory_node: runs BEFORE the Planner to inject conversation context.
  - save_memory_node: runs AFTER the Evidence Synthesis Agent to persist the turn.
"""

from __future__ import annotations

import logging

from app.graph.state import ClinicalState
from app.memory.redis_client import load_session, save_session

logger = logging.getLogger(__name__)


def load_memory_node(state: ClinicalState) -> dict:
    """
    Load session memory from Redis and inject it into the graph state.
    """
    session_id = state.get("session_id", "default")
    logger.info("[MemoryAgent] Loading session context for session_id=%s", session_id)

    memory_context = load_session(session_id)
    logger.debug(
        "[MemoryAgent] Loaded %d history turns",
        len(memory_context.get("conversation_history", [])),
    )
    return {"memory_context": memory_context}


def save_memory_node(state: ClinicalState) -> dict:
    """
    Save the current turn to Redis after the Evidence Synthesis Agent completes.
    """
    session_id = state.get("session_id", "default")
    query = state.get("query", "")
    final_response = state.get("final_response")

    if not final_response:
        logger.warning("[MemoryAgent] No final_response to save for session_id=%s", session_id)
        return {}

    summary = final_response.summary
    plan = state.get("plan")
    if hasattr(plan, "clinical_topic"):
        clinical_topic = plan.clinical_topic or ""
    elif isinstance(plan, dict):
        clinical_topic = plan.get("clinical_topic", "")
    else:
        clinical_topic = ""

    # Collect evidence references (PMIDs and PDF names from citations)
    evidence_refs: list[str] = []
    for citation in final_response.citations:
        if citation.pmid:
            evidence_refs.append(f"PMID:{citation.pmid}")
        if citation.source_type == "clinical_guideline" and citation.title:
            evidence_refs.append(citation.title)

    save_session(
        session_id=session_id,
        query=query,
        summary=summary,
        clinical_topic=clinical_topic,
        evidence_refs=evidence_refs,
    )

    logger.info("[MemoryAgent] Session saved for session_id=%s", session_id)
    return {}

"""
Clinical Reasoning Agent.

Interprets ranked evidence, compares findings, detects agreements and
disagreements, and produces a structured reasoning narrative.
Never retrieves documents; doesn't search Chroma or PubMed; doesn't invent an answer.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ClinicalState, RankedEvidence
from app.llm.factory import get_llm
from app.prompts import CLINICAL_REASONING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _format_ranked_evidence(ranked_evidence: list[RankedEvidence]) -> str:
    """Serialise classified evidence for the LLM prompt.
       It takes every single item from ranked_evidence (both Guidelines and PubMed articles) and serializes them into JSON.    
    """
    items = []
    for item in ranked_evidence:
        items.append(
            {
                "rank": item.rank,
                "evidence_type": item.evidence_type,
                "source_type": item.source_type,
                "title": item.title,
                "text": item.text[:3000],  # allow full chunk text
                "source_ref": item.source_ref,
                "collection": item.collection,
            }
        )
    return json.dumps(items, indent=2)


def run_clinical_reasoning(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Clinical Reasoning Agent.

    Reads:  query, ranked_evidence, evidence_support
    Writes: reasoning
    """
    query = state.get("query", "")
    ranked_evidence = state.get("ranked_evidence", [])
    evidence_support = state.get("evidence_support")

    if not ranked_evidence:
        logger.warning("[ClinicalReasoningAgent] No evidence retrieved for query.")
        return {
            "reasoning": (
                "No retrieved clinical evidence items were available for this query. "
                "Unable to perform an evidence-grounded analysis without retrieved sources."
            )
        }


    logger.info(
        "[ClinicalReasoningAgent] Reasoning over %d classified evidence items …",
        len(ranked_evidence),
    )

    evidence_str = _format_ranked_evidence(ranked_evidence)
    llm = get_llm(temperature=0.0)


    support_summary = (
        evidence_support.retrieval_relevance_summary if evidence_support else "Retrieved evidence items loaded."
    )

    messages = [
        SystemMessage(content=CLINICAL_REASONING_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical question:\n{query}\n\n"
                f"Evidence Retrieval Support Summary: {support_summary}\n\n"
                f"Classified Evidence Items:\n{evidence_str}"
            )
        ),
    ]

    response = llm.invoke(messages)
    reasoning = response.content.strip()

    logger.info(
        "[ClinicalReasoningAgent] Reasoning produced (%d chars)", len(reasoning)
    )
    return {"reasoning": reasoning}


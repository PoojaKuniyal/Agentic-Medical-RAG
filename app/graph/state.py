"""
Shared LangGraph state definition.

Defines the single TypedDict state object (ClinicalState) that flows through
all nodes and conditional edges in the LangGraph workflow.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from app.graph.schemas import (
    ClinicalSummaryResponse,
    EvidenceSupport,
    GuidelineChunk,
    PlannerOutput,
    PubMedArticle,
    RankedEvidence,
    SafetyResponse,
)

__all__ = ["ClinicalState"]


class ClinicalState(TypedDict, total=False):
    """
    The single shared state object that flows through the entire LangGraph.

    Fields are marked Optional/total=False because nodes only write the
    fields they are responsible for; earlier nodes will not have set
    later fields yet.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query: str
    session_id: str

    # ── Guardrail ─────────────────────────────────────────────────────────
    is_safe: bool
    safety_response: Optional[SafetyResponse]

    # ── Memory (loaded before Planner) ────────────────────────────────────
    memory_context: dict[str, Any]

    # ── Planner ───────────────────────────────────────────────────────────
    plan: Optional[PlannerOutput]

    # ── Retrieval (populated in parallel) ─────────────────────────────────
    guideline_evidence: list[GuidelineChunk]
    pubmed_evidence: list[PubMedArticle]
    pubmed_query: Optional[str]
    pubmed_error_reason: Optional[str]

    # ── Evidence Classification & Support Metrics ─────────────────────────
    ranked_evidence: list[RankedEvidence]
    evidence_support: Optional[EvidenceSupport]

    # ── Clinical Reasoning ────────────────────────────────────────────────
    reasoning: str

    # ── Reflection ────────────────────────────────────────────────────────
    reflection_needed: bool
    reflection_count: int               # hard cap: MAX_REFLECTION_ITERATIONS

    # ── Summary ───────────────────────────────────────────────────────────
    final_response: Optional[ClinicalSummaryResponse]

    # ── Observability & Tracing ───────────────────────────────────────────
    langsmith_run_id: Optional[str]
    planner_intent: Optional[str]
    retrieval_agents_selected: list[str]
    chroma_collections_selected: list[str]
    guideline_results_count: int
    pubmed_results_count: int
    filtered_pubmed_results_count: int

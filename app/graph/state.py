"""
Shared LangGraph state schema and all structured output schemas.

This is the single source of truth for data flowing through the graph.
Every agent reads from and writes to ClinicalState.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Structured output schemas (used by agents and API responses)

class GuidelineChunk(BaseModel):
    """A single retrieved chunk from the Chroma vector store."""

    chunk_id: str
    text: str
    source_pdf: str                         # filename of the source PDF
    page: Optional[int] = None
    collection: str                         # Chroma collection name
    score: float = 0.0                      # cosine similarity score


class PubMedArticle(BaseModel):
    """A single article record retrieved from PubMed."""

    pmid: str
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: str = ""
    publication_types: list[str] = Field(default_factory=list)
    # e.g. ["Randomized Controlled Trial", "Meta-Analysis"]
    url: str = ""


class RankedEvidence(BaseModel):
    """A single evidence item after deterministic classification and ordering by retrieval relevance."""

    rank: int
    evidence_type: str                      # e.g. "Clinical Guideline", "Systematic Review", "Other / Unclassified"
    source_type: str                        # "guideline" | "pubmed"
    title: str
    text: str
    source_ref: str                         # PDF filename or PMID
    collection: Optional[str] = None


class EvidenceSupport(BaseModel):
    """Objective summary of retrieved evidence metrics (replaces fabricated confidence score)."""

    guideline_passages_count: int = 0
    pubmed_studies_count: int = 0
    total_sources_count: int = 0
    study_type_distribution: dict[str, int] = Field(default_factory=dict)
    publication_year_range: Optional[str] = None
    retrieval_relevance_summary: str = ""


class EvidenceSource(BaseModel):
    """High-level evidence source summary for the final response."""

    source_type: str = "clinical_guideline" # "clinical_guideline" | "pubmed"
    collection: Optional[str] = None
    title: str = ""
    evidence_type: str = "Clinical Guideline"


class Citation(BaseModel):
    """A formatted citation for the final response."""

    reference_id: str                       # e.g. "REF-001"
    source_type: str                        # "clinical_guideline" | "pubmed"
    title: str
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    page: Optional[int] = None
    chunk_text: Optional[str] = None


class PlannerOutput(BaseModel):
    """The Planner Agent's routing decision."""

    intent: str
    clinical_topic: str  # e.g., "type 2 diabetes", "hypertension"
    retrieval_agents: list[str]             # e.g. ["guideline_rag", "pubmed"]
    chroma_collections: list[str]           # e.g. ["clinical_guidelines"]
    reasoning: str


from pydantic import BaseModel, Field, field_validator


class GuardrailDecision(BaseModel):
    """Structured decision output from the Guardrail Agent."""

    is_safe: bool = Field(
        description="True if query is safe to process into clinical evidence synthesis, False if blocked."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for blocking if unsafe (e.g., diagnosis, emergency, unsafe, prompt_injection, misinformation).",
    )
    message: Optional[str] = Field(
        default=None,
        description="User-facing explanation if unsafe.",
    )

    @field_validator("is_safe", mode="before")
    @classmethod
    def parse_is_safe(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)


class SafetyResponse(BaseModel):
    """Returned when the Guardrail Agent rejects a query."""

    blocked: bool = True
    reason: str
    message: str



class ClinicalSummaryResponse(BaseModel):
    """The final structured response returned to the user."""

    summary: str
    evidence_support: EvidenceSupport = Field(default_factory=EvidenceSupport)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


# Shared LangGraph state

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



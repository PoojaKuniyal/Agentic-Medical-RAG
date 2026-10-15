"""
Evidence Classification & Organization Node. 

Deterministically classifies evidence study/source types using explicit metadata
and organizes items by retrieval relevance without LLM inference or fabricated hierarchy.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.graph.schemas import (
    EvidenceSupport,
    GuidelineChunk,
    PubMedArticle,
    RankedEvidence,
)
from app.graph.state import ClinicalState

logger = logging.getLogger(__name__)

# Allowed evidence study/source types
TYPE_CLINICAL_GUIDELINE = "Clinical Guideline"
TYPE_CONSENSUS_REPORT = "Consensus Report"
TYPE_SYSTEMATIC_REVIEW = "Systematic Review"
TYPE_META_ANALYSIS = "Meta-analysis"
TYPE_RCT = "Randomized Controlled Trial"
TYPE_OBSERVATIONAL_STUDY = "Observational Study"
TYPE_OTHER_UNCLASSIFIED = "Other / Unclassified"

# PubMed publication_type keyword mappings (checked in order of specificity)
PUBMED_TYPE_MAP: list[tuple[str, str]] = [
    ("systematic review", TYPE_SYSTEMATIC_REVIEW),
    ("meta-analysis", TYPE_META_ANALYSIS),
    ("randomized controlled trial", TYPE_RCT),
    ("cohort study", TYPE_OBSERVATIONAL_STUDY),
    ("case-control study", TYPE_OBSERVATIONAL_STUDY),
    ("observational study", TYPE_OBSERVATIONAL_STUDY),
]


def _classify_guideline_chunk(chunk: GuidelineChunk) -> str:
    """Classify local PDF chunk type strictly based on explicit collection metadata."""
    collection = (chunk.collection or "").lower()
    if collection == "clinical_guidelines":
        return TYPE_CLINICAL_GUIDELINE
    elif collection == "consensus_reports":
        return TYPE_CONSENSUS_REPORT
    return TYPE_OTHER_UNCLASSIFIED


def _classify_pubmed_article(article: PubMedArticle) -> str:
    """Classify PubMed article type strictly based on official publication_types metadata."""
    
    for pub_type in article.publication_types:
        pub_lower = pub_type.lower()
        for keyword, mapped_type in PUBMED_TYPE_MAP:
            if keyword in pub_lower:
                return mapped_type
    return TYPE_OTHER_UNCLASSIFIED


from app.config import get_settings
from app.rag.reranker import rerank_candidates


def run_evidence_ranking(state: ClinicalState) -> dict:
    """
    LangGraph node function for Evidence Classification & Organization.
    Applies Cross-Encoder re-ranking across both Guidelines and PubMed articles
    when enabled, falling back to deterministic metadata-based ordering if disabled.
    """
    query = state.get("query", "")
    guideline_evidence = state.get("guideline_evidence", [])
    pubmed_evidence = state.get("pubmed_evidence", [])
    settings = get_settings()

    candidate_items: list[dict] = []
    type_counts: dict[str, int] = {}
    years: list[int] = []

    # 1. Process Guideline Chunks
    sorted_guidelines = sorted(guideline_evidence, key=lambda c: c.score, reverse=True)
    for chunk in sorted_guidelines:
        evidence_type = _classify_guideline_chunk(chunk)
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        candidate_items.append(
            {
                "evidence_type": evidence_type,
                "source_type": "guideline",
                "title": chunk.source_pdf,
                "text": chunk.text,
                "source_ref": chunk.source_pdf,
                "collection": chunk.collection,
                "score": chunk.score,
            }
        )

    # 2. Process PubMed Articles
    for article in pubmed_evidence:
        evidence_type = _classify_pubmed_article(article)
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        if article.year:
            years.append(article.year)

        candidate_items.append(
            {
                "evidence_type": evidence_type,
                "source_type": "pubmed",
                "title": article.title,
                "text": f"{article.title}. {article.abstract}" if article.title else article.abstract,
                "source_ref": article.pmid,
                "collection": None,
                "score": 0.0,
            }
        )

    # 3. Re-ranking: Cross-Encoder (if enabled)
    reranked_applied = False
    if settings.enable_reranking and candidate_items and query.strip():
        try:
            candidate_items = rerank_candidates(
                query=query,
                candidates=candidate_items,
                text_key="text",
                top_k=settings.reranker_top_k,
            )
            reranked_applied = True
            logger.info(
                "[EvidenceRanking] Applied Cross-Encoder reranking (%s) on %d candidates",
                settings.reranker_model,
                len(candidate_items),
            )
        except Exception as exc:
            logger.warning(
                "[EvidenceRanking] Cross-Encoder reranking failed (%s). Using standard ordering.",
                exc,
            )

    # Build final RankedEvidence models with sequential ranks 1..N
    items: list[RankedEvidence] = []
    for index, item in enumerate(candidate_items, 1):
        items.append(
            RankedEvidence(
                rank=index,
                evidence_type=item["evidence_type"],
                source_type=item["source_type"],
                title=item["title"],
                text=item["text"],
                source_ref=item["source_ref"],
                collection=item.get("collection"),
            )
        )

    # Build year range string
    year_range_str: Optional[str] = None
    if years:
        min_y, max_y = min(years), max(years)
        year_range_str = f"{min_y}" if min_y == max_y else f"{min_y}–{max_y}"

    # Build objective relevance summary
    rerank_mode_str = f"Re-ranked via {settings.reranker_model}" if reranked_applied else "Deterministic similarity order"
    relevance_summary = (
        f"{len(guideline_evidence)} guideline passage(s) retrieved via vector search; "
        f"{len(pubmed_evidence)} PubMed study record(s) retrieved. "
        f"Final ranking ({rerank_mode_str}): {len(items)} evidence item(s) selected."
    )

    support_metrics = EvidenceSupport(
        guideline_passages_count=len(guideline_evidence),
        pubmed_studies_count=len(pubmed_evidence),
        total_sources_count=len(items),
        study_type_distribution=type_counts,
        publication_year_range=year_range_str,
        retrieval_relevance_summary=relevance_summary,
    )

    logger.info(
        "[EvidenceClassification] Finalized %d ranked evidence items (%d guidelines, %d pubmed, reranked=%s)",
        len(items),
        len(guideline_evidence),
        len(pubmed_evidence),
        reranked_applied,
    )

    return {
        "ranked_evidence": items,
        "evidence_support": support_metrics,
    }


# 1. Candidate Generation: Gathers candidate chunks from Chroma and candidate articles from PubMed.
# 2. Cross-Encoder Re-Ranking: When enabled, passes (query, candidate_text) pairs to the Cross-Encoder model.
# 3. Unified Relevance: Cross-evaluates Guidelines and PubMed studies on the same semantic scale, putting the most clinically relevant evidence at Rank #1.
# 4. Fallback: Gracefully falls back to deterministic ordering if reranking is disabled or unavailable.

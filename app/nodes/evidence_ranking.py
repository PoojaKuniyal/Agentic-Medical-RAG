"""
Evidence Classification & Organization Node. 

Deterministically classifies evidence study/source types using explicit metadata
and organizes items by retrieval relevance without LLM inference or fabricated hierarchy.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.graph.state import (
    ClinicalState,
    EvidenceSupport,
    GuidelineChunk,
    PubMedArticle,
    RankedEvidence,
)

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


def _classify_pubmed_article(publication_types: list[str]) -> str:
    """Classify PubMed article type strictly based on official publication_types metadata."""
    
    for pub_type in publication_types:
        pub_lower = pub_type.lower()
        for keyword, mapped_type in PUBMED_TYPE_MAP:
            if keyword in pub_lower:
                return mapped_type
    return TYPE_OTHER_UNCLASSIFIED


def run_evidence_ranking(state: ClinicalState) -> dict:
    """
    LangGraph node function for Evidence Classification & Organization.

    Reads:  guideline_evidence, pubmed_evidence
    Writes: ranked_evidence, evidence_support
    """
    guideline_evidence = state.get("guideline_evidence", [])
    pubmed_evidence = state.get("pubmed_evidence", [])

    items: list[RankedEvidence] = []
    type_counts: dict[str, int] = {}
    years: list[int] = []

    # Process Guideline Chunks (preserve vector search score order)
    # Already sorted, but re-sort even if guideline_evidence came from a state checkpointer, fallback logic, or was modified out-of-order,
    # evidence_ranking_agent would strictly process chunks in similarity score order.   
    sorted_guidelines = sorted(guideline_evidence, key=lambda c: c.score, reverse=True)
    for chunk in sorted_guidelines:
        evidence_type = _classify_guideline_chunk(chunk)
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        items.append(
            RankedEvidence(             # transforms retrieval-specific GuidelineChunk into a more general evidence representation
                rank=len(items) + 1,
                evidence_type=evidence_type,
                source_type="guideline",
                title=chunk.source_pdf,
                text=chunk.text,
                source_ref=chunk.source_pdf,
                collection=chunk.collection,
            )
        )

    # 2. Process PubMed Articles
    for article in pubmed_evidence:
        evidence_type = _classify_pubmed_article(article.publication_types)
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        if article.year:
            years.append(article.year)

        items.append(
            RankedEvidence(
                rank=len(items) + 1,
                evidence_type=evidence_type,
                source_type="pubmed",
                title=article.title,
                text=article.abstract,
                source_ref=article.pmid,
                collection=None,
            )
        )

    # Re-assign sequential rank 1..N
    for index, item in enumerate(items, 1):
        item.rank = index

    # Build year range string
    year_range_str: Optional[str] = None
    if years:
        min_y, max_y = min(years), max(years)
        year_range_str = f"{min_y}" if min_y == max_y else f"{min_y}–{max_y}"

    # Build objective relevance summary
    relevance_summary = (
        f"{len(guideline_evidence)} guideline passage(s) retrieved via vector search; "
        f"{len(pubmed_evidence)} PubMed study record(s) retrieved."
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
        "[EvidenceClassification] Processed %d total evidence items deterministically (%d guidelines, %d pubmed)",
        len(items),
        len(guideline_evidence),
        len(pubmed_evidence),
    )

    return {
        "ranked_evidence": items,
        "evidence_support": support_metrics,
    }

# Guidelines are placed at the top of the list (sorted). 
# PubMed are appended after the guidelines, preserving their retrieval sequence. 
# Sequential Ranks Re-assigned : After all items are collected, their ranks are reset to 1..N. 
# It Doesnot skip any evidence. Every retrieved guideline chunk and PubMed article is included in ranked_evidence.  
# Although Guidelines are given higher rank it does NOT skip good answers from PubMed in the clinicial reasoning stage. 
# The LLM reads all items, so any good answer or evidence found in a PubMed article is fully available to the LLM during the clinical reasoning stage.

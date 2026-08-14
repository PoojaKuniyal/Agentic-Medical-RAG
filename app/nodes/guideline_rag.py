"""
Clinical Guideline RAG Retrieval Step — Chroma retrieval only.

Queries the specified Chroma collections using the clinical query.
Performs pure database retrieval. 
"""

from __future__ import annotations

import logging

from app.graph.state import ClinicalState, GuidelineChunk
from app.rag.vectorstore import similarity_search

logger = logging.getLogger(__name__)

DEFAULT_N_RESULTS = 10 # Number of chunks to retrieve from each collection


def run_guideline_rag(state: ClinicalState) -> dict:
    """
    LangGraph node function for clinical guideline retrieval.

    Reads:  query, plan (for chroma_collections)
    Writes: guideline_evidence, guideline_results_count
    """
    query = state["query"]
    plan = state.get("plan") 
        # Gets the Planner's collection decision
    collections: list[str] = ( 
        plan.chroma_collections
        if plan and plan.chroma_collections
        else ["clinical_guidelines", "consensus_reports", "research_articles"]
    )

    logger.info(
        "[GuidelineRAGAgent] Searching collections=%s for query='%s'",
        collections,
        query[:80],
    )

    all_chunks: list[GuidelineChunk] = []

    for collection_name in collections:
        try:
            raw_chunks = similarity_search(
                collection_name=collection_name,
                query=query,
                n_results=DEFAULT_N_RESULTS,
            )
            # Converts raw Chroma results into GuidelineChunk Pydantic model objects 
            for chunk in raw_chunks: 
                all_chunks.append( # combines results from all collections
                    GuidelineChunk(       
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        source_pdf=chunk["source_pdf"],
                        page=chunk.get("page_number"),
                        collection=chunk["collection"],
                        score=chunk["score"],
                    )
                )
        except Exception as exc:
            logger.error(
                "[GuidelineRAGAgent] Error querying collection '%s': %s",
                collection_name,
                exc,
            )

    # Sort by score descending: chunks that were most semantically similar to the query 
    all_chunks.sort(key=lambda c: c.score, reverse=True)

    logger.info(
        "[GuidelineRAGAgent] Retrieved %d chunks from %d collections",
        len(all_chunks),
        len(collections),
    )
    return {
        "guideline_evidence": all_chunks,
        # records the observability count of the chunks for LangSmith
        "guideline_results_count": len(all_chunks),     
    }

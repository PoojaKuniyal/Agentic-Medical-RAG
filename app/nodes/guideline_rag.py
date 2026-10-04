"""
Clinical Guideline RAG Retrieval Step — Chroma retrieval only.

Queries the specified Chroma collections using the clinical query.
Performs pure database retrieval. 
"""

from __future__ import annotations

import logging

from app.graph.schemas import GuidelineChunk
from app.graph.state import ClinicalState
from app.rag.vectorstore import similarity_search

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_N_RESULTS = 5  # Number of top chunks to retrieve from each collection
MAX_TOTAL_GUIDELINE_CHUNKS = 10  # Maximum overall top-scoring chunks to retain


def _is_boilerplate_chunk(text: str) -> bool:
    """Filter out short header/footer notices, page numbers, and copyright boilerplate."""
    cleaned = text.strip()
    if len(cleaned) < 80:
        return True
    lower = cleaned.lower()
    if "notice-of-rights" in lower or "terms-and-conditions" in lower:
        return True
    return False


def run_guideline_rag(state: ClinicalState) -> dict:
    """
    LangGraph node function for local PDF knowledge retrieval.
    """
    query = state["query"]
    plan = state.get("plan") 
    settings = get_settings()
    all_known_collections = list(settings.chroma_collections.values())

    # Search all local knowledge collections by default if guideline_rag is invoked
    collections: list[str] = ( 
        plan.chroma_collections
        if plan and plan.chroma_collections
        else all_known_collections
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
                text_content = chunk["text"]
                if _is_boilerplate_chunk(text_content):
                    continue
                all_chunks.append(
                    GuidelineChunk(       
                        chunk_id=chunk["chunk_id"],
                        text=text_content,
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

    # Sort by score descending and keep only the top N highest-quality chunks
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    all_chunks = all_chunks[:MAX_TOTAL_GUIDELINE_CHUNKS]

    logger.info(
        "[GuidelineRAGAgent] Retained top %d high-relevance chunks from %d collections",
        len(all_chunks),
        len(collections),
    )
    return {
        "guideline_evidence": all_chunks,
        # records the observability count of the chunks for LangSmith
        "guideline_results_count": len(all_chunks),     
    }

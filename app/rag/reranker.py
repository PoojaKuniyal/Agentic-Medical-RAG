"""
Cross-Encoder Reranker for Clinical RAG.

Uses a cross-encoder model to score (query, document) pairs directly
with deep cross-attention, providing high-precision relevance ranking
across both local clinical guidelines and PubMed literature.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker():
    """Load and cache the CrossEncoder model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install it with: pip install sentence-transformers"
        ) from e

    settings = get_settings()
    logger.info("Loading Cross-Encoder reranker model: %s", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    logger.info("Cross-Encoder reranker model loaded.")
    return model


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    text_key: str = "text",
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank a list of candidate dictionaries based on semantic relevance to the query.
    """
    if not candidates:
        return []

    settings = get_settings()
    k = top_k if top_k is not None else settings.reranker_top_k

    # If only 1 candidate or empty query, return with default score
    if len(candidates) == 1 or not query.strip():
        scored = list(candidates)
        for item in scored:
            if "rerank_score" not in item:
                item["rerank_score"] = item.get("score", 1.0)
        return scored[:k]

    try:
        reranker = _get_reranker()
        pairs = [[query, item[text_key]] for item in candidates]
        scores = reranker.predict(pairs)

        # Attach rerank score to each item
        scored_candidates = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            scored_candidates.append(enriched)

        # Sort descending by rerank_score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:k]

    except Exception as exc:
        logger.warning(
            "[Reranker] Cross-Encoder reranking failed (%s). Falling back to original order.",
            exc,
        )
        return candidates[:k]

"""
Sentence Transformers embedder.

Uses the model specified by EMBEDDING_MODEL in .env (default: all-MiniLM-L6-v2).
The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache # to cache the model and only load it once

from app.config import get_settings 

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1) 
def _get_model():
    """Load and cache the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from e

    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Parameters
    ----------
    texts : The texts to embed.

    Returns
    -------
    list of embedding vectors (each a list of floats).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate a single query embedding."""
    return embed_texts([query])[0]

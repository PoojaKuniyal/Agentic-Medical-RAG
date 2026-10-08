"""
Chroma multi-collection client.
Supports two modes (configured via CHROMA_MODE in .env):
  • http   — connects to a running Chroma HTTP server (production/Docker)
  • local  — uses a persistent local directory (development)
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

import chromadb
from chromadb import Collection

from app.config import get_settings
from app.rag.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

# Similarity Threshold Filter - to throw away non-matching chunks. 
MIN_SCORE_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    """Return a cached singleton Chroma client."""
    settings = get_settings()

    if settings.chroma_mode == "http":
        logger.info(
            "Connecting to Chroma HTTP server at %s:%d",
            settings.chroma_host,
            settings.chroma_port,
        )
        return chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    else:
        logger.info(
            "Using local persistent Chroma at %s", settings.chroma_persist_dir
        )
        return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(name: str) -> Collection:
    """
    Get or create a named Chroma collection.
    Uses cosine similarity (appropriate for sentence-transformer embeddings).
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Collection '%s' ready (%d chunks)", name, collection.count())
    return collection


def upsert_chunks(collection_name: str, chunks: list[dict]) -> int:
    """
    Upsert a list of text chunks into the specified Chroma collection.
    """
    if not chunks:
        return 0

    collection = get_collection(collection_name)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    ids = [
        f"{c['source_pdf']}__p{c['page_number']}__c{c['chunk_index']}__{uuid.uuid4().hex[:8]}"
        for c in chunks
    ]
    metadatas = [
        {
            "source_pdf": c["source_pdf"],
            "page_number": c["page_number"],
            "collection": c["collection"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into collection '%s'", len(chunks), collection_name)
    return len(chunks)


def similarity_search(
    collection_name: str,
    query: str,
    n_results: int = 8,
    min_score:float = MIN_SCORE_THRESHOLD, # filters out low relevance noise
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query.
    """
    collection = get_collection(collection_name)
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # Chroma cosine distance or similarity score (1 - distance)
        score = max(0.0, 1.0 - dist)
        if score >= min_score:  # Only accept reasonably matching chunks
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source_pdf": meta.get("source_pdf", ""),
                    "page_number": meta.get("page_number"),
                    "collection": meta.get("collection", collection_name),
                    "score": round(score, 4),
                }
            )

    return chunks


def delete_collection(name: str) -> None:
    """Delete a named collection (used for force rebuild)."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
        logger.info("Deleted collection '%s'", name)
    except Exception as exc:
        logger.warning("Could not delete collection '%s': %s", name, exc)


def delete_chunks_by_source_pdf(collection_name: str, source_pdf: str) -> None:
    """
    Delete all chunks in a Chroma collection belonging to a specific source PDF.
    """
    collection = get_collection(collection_name)
    try:
        collection.delete(where={"source_pdf": source_pdf})
        logger.info(
            "Deleted existing Chroma chunks for '%s' in collection '%s'",
            source_pdf,
            collection_name,
        )
    except Exception as exc:
        logger.warning(
            "Could not delete chunks for '%s' in collection '%s': %s",
            source_pdf,
            collection_name,
            exc,
        )


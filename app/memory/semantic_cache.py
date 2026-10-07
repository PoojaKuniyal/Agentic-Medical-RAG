"""
Caches verified clinical responses and safety decisions by vector similarity.
If a new user query is semantically equivalent to a previously answered query
(cosine similarity >= threshold), the cached result is returned immediately,
bypassing the multi-agent pipeline and saving LLM costs & latency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from app.config import get_settings
from app.rag.embedder import embed_query
from app.rag.vectorstore import get_chroma_client

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_COLLECTION = "semantic_cache"


class SemanticCache:
    """
    Vector-similarity semantic cache for clinical RAG responses.
    Synchronizes Chroma embeddings with Redis TTL via metadata expiration timestamps,
    top-K candidate retrieval (to prevent stale-entry occlusion), and automatic lazy/batch purging.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()  # lock to ensure thread safety when updating cache metrics
        self._hits = 0  # total number of times we got the answer from semantic cache
        self._misses = 0  # total number of times we didn't get the answer from semantic cache
        self._fallback_store: dict[str, dict[str, Any]] = {}  # temporary storage if Redis unavailable: {id: {"payload": ..., "expires_at": ...}}

    def _get_collection(self):
        """Get or create the Chroma collection for semantic caching."""
        client = get_chroma_client()
        return client.get_or_create_collection(
            name=SEMANTIC_CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _cache_key(self, cache_id: str) -> str:
        """Redis key for the cached payload."""
        return f"mediai:semantic_cache:{cache_id}"

    def _generate_cache_id(self, query: str) -> str:
        """Generate a deterministic 16-character hash ID for a query string."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @traceable(name="SemanticCache.lookup", run_type="retriever")
    def lookup(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 3,
    ) -> Optional[dict[str, Any]]:
        """
        Search for a semantically similar cached query.
        Retrieves top-K candidates to prevent expired top-1 entries from blocking valid matches.
        """
        if not self.settings.enable_semantic_cache:
            return None

        sim_threshold = threshold if threshold is not None else self.settings.semantic_cache_threshold
        now = time.time()

        try:
            collection = self._get_collection()
            count = collection.count()
            if count == 0:
                with self._lock:
                    self._misses += 1
                return None

            query_embedding = embed_query(query)
            n_results = min(top_k, count)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not ids:
                with self._lock:
                    self._misses += 1
                return None

            # Iterate through top-k candidates in descending similarity order
            for i in range(len(ids)):
                candidate_id = ids[i]
                matched_query = documents[i] if i < len(documents) else ""
                distance = distances[i] if i < len(distances) else 1.0
                metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

                # Chroma cosine distance: d = 1 - cos(theta) -> similarity = 1 - d
                similarity = max(0.0, 1.0 - distance)

                # If the highest remaining similarity is below threshold, stop evaluating
                if similarity < sim_threshold:
                    logger.debug(
                        "[SemanticCache] Candidate query '%s' similarity %.3f < threshold %.3f (no further match)",
                        matched_query[:50],
                        similarity,
                        sim_threshold,
                    )
                    break

                # 1. Check TTL metadata in Chroma (fast filter)
                expires_at = metadata.get("expires_at")
                if expires_at is not None and expires_at < now:
                    logger.debug(
                        "[SemanticCache] Candidate cache_id=%s expired by metadata (expires_at=%.0f, now=%.0f), deleting from Chroma",
                        candidate_id,
                        expires_at,
                        now,
                    )
                    try:
                        collection.delete(ids=[candidate_id])
                    except Exception:
                        pass
                    continue  # Check next candidate

                # 2. Retrieve payload from Redis or in-memory fallback (definitive check)
                payload = self._load_payload(candidate_id)

                if not payload:
                    logger.debug(
                        "[SemanticCache] Payload expired or missing in Redis for cache_id=%s, deleting stale Chroma vector",
                        candidate_id,
                    )
                    # Clean up stale vector entry in Chroma
                    try:
                        collection.delete(ids=[candidate_id])
                    except Exception:
                        pass
                    continue  # Check next candidate

                # Valid cache hit!
                with self._lock:
                    self._hits += 1

                logger.info(
                    "[SemanticCache] HIT! Matched query: '%s' (similarity: %.3f >= %.3f) [cache_id=%s]",
                    matched_query[:60],
                    similarity,
                    sim_threshold,
                    candidate_id,
                )

                return {
                    "response": payload.get("response"),
                    "safety_response": payload.get("safety_response"),
                    "is_safe": payload.get("is_safe", True),
                    "similarity": round(similarity, 4),
                    "matched_query": matched_query,
                    "cache_id": candidate_id,
                }

            # If no candidate matched and was valid
            with self._lock:
                self._misses += 1
            return None

        except Exception as exc:
            logger.warning("[SemanticCache] Lookup failed (falling back to live pipeline): %s", exc)
            with self._lock:
                self._misses += 1
            return None

    @traceable(name="SemanticCache.store", run_type="tool")
    def store(
        self,
        query: str,
        response: Optional[dict[str, Any] | BaseModel] = None,
        is_safe: bool = True,
        safety_response: Optional[dict[str, Any] | BaseModel] = None,
        session_id: str = "",
    ) -> Optional[str]:
        """
        Store a query and its resulting response in the semantic cache with matching TTL in Redis & Chroma.
        """
        if not self.settings.enable_semantic_cache:
            return None

        # Do not cache empty responses
        if response is None and safety_response is None:
            return None

        try:
            cache_id = self._generate_cache_id(query)
            now = time.time()
            ttl = self.settings.semantic_cache_ttl
            expires_at = now + ttl

            # Serialize models if needed
            resp_data = (
                response.model_dump()
                if isinstance(response, BaseModel)
                else response
            )
            safety_data = (
                safety_response.model_dump()
                if isinstance(safety_response, BaseModel)
                else safety_response
            )

            payload = {
                "cache_id": cache_id,
                "query": query,
                "is_safe": is_safe,
                "response": resp_data,
                "safety_response": safety_data,
                "session_id": session_id,
                "created_at": now,
                "expires_at": expires_at,
            }

            # 1. Store payload in Redis or in-memory fallback with TTL
            self._save_payload(cache_id, payload, expires_at=expires_at)

            # 2. Store vector embedding in Chroma with expiration metadata
            collection = self._get_collection()
            query_embedding = embed_query(query)

            collection.upsert(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[
                    {
                        "cache_id": cache_id,
                        "created_at": now,
                        "expires_at": expires_at,
                        "is_safe": int(is_safe),
                    }
                ],
            )

            logger.info(
                "[SemanticCache] Stored query '%s' [cache_id=%s, ttl=%ds]",
                query[:60],
                cache_id,
                ttl,
            )
            return cache_id

        except Exception as exc:
            logger.warning("[SemanticCache] Failed to store cache entry: %s", exc)
            return None

    def _save_payload(
        self,
        cache_id: str,
        payload: dict[str, Any],
        expires_at: Optional[float] = None,
    ) -> None:
        """Save payload to Redis with TTL, falling back to memory store."""
        key = self._cache_key(cache_id)
        raw = json.dumps(payload)
        ttl = self.settings.semantic_cache_ttl
        exp = expires_at if expires_at is not None else (time.time() + ttl)

        try:
            from app.memory.redis_client import get_redis
            client = get_redis()
            client.set(key, raw, ex=ttl)
        except Exception as exc:
            logger.debug("[SemanticCache] Redis store unavailable (%s), using memory fallback", exc)
            with self._lock:
                self._fallback_store[cache_id] = {
                    "payload": payload,
                    "expires_at": exp,
                }

    def _load_payload(self, cache_id: str) -> Optional[dict[str, Any]]:
        """Load payload from Redis or in-memory fallback with TTL validation."""
        key = self._cache_key(cache_id)

        try:
            from app.memory.redis_client import get_redis
            client = get_redis()
            raw = client.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.debug("[SemanticCache] Redis load error (%s), checking memory fallback", exc)

        with self._lock:
            entry = self._fallback_store.get(cache_id)
            if entry:
                # Validate in-memory TTL
                if entry.get("expires_at", float("inf")) < time.time():
                    del self._fallback_store[cache_id]
                    return None
                return entry.get("payload")

        return None

    def purge_expired(self) -> int:
        """
        Delete all expired vector entries from Chroma and memory fallback store.
        """
        purged = 0
        now = time.time()
        ttl = self.settings.semantic_cache_ttl

        # 1. Clean Chroma collection
        try:
            collection = self._get_collection()
            count = collection.count()
            if count > 0:
                try:
                    expired_records = collection.get(
                        where={"expires_at": {"$lt": now}},
                        include=["metadatas"],
                    )
                    expired_ids = expired_records.get("ids", [])
                    if expired_ids:
                        collection.delete(ids=expired_ids)
                        purged += len(expired_ids)
                        logger.info(
                            "[SemanticCache] Purged %d expired vector entries from Chroma",
                            len(expired_ids),
                        )
                except Exception as query_exc:
                    logger.debug(
                        "[SemanticCache] Chroma where query failed, falling back to metadata scan: %s",
                        query_exc,
                    )
                    all_records = collection.get(include=["metadatas"])
                    ids_to_delete = []
                    for cid, meta in zip(
                        all_records.get("ids", []),
                        all_records.get("metadatas", []),
                    ):
                        if not meta:
                            continue
                        exp = meta.get("expires_at")
                        if exp is not None and exp < now:
                            ids_to_delete.append(cid)
                        elif exp is None:
                            created = meta.get("created_at", 0)
                            if created and (now - created) > ttl:
                                ids_to_delete.append(cid)

                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)
                        purged += len(ids_to_delete)
                        logger.info(
                            "[SemanticCache] Purged %d expired vector entries from Chroma by scan",
                            len(ids_to_delete),
                        )
        except Exception as exc:
            logger.warning("[SemanticCache] Error purging expired Chroma entries: %s", exc)

        # 2. Clean fallback store
        with self._lock:
            expired_fallback_keys = [
                k
                for k, v in self._fallback_store.items()
                if v.get("expires_at", float("inf")) < now
            ]
            for k in expired_fallback_keys:
                del self._fallback_store[k]
                purged += 1

        return purged

    def clear(self) -> int:
        """
        Clear all entries from the semantic cache.
        """
        cleared_count = 0
        try:
            collection = self._get_collection()
            cleared_count = collection.count()
            client = get_chroma_client()
            client.delete_collection(SEMANTIC_CACHE_COLLECTION)
            logger.info("[SemanticCache] Cleared %d vector entries from Chroma", cleared_count)
        except Exception as exc:
            logger.warning("[SemanticCache] Failed to clear Chroma collection: %s", exc)

        # Clear Redis keys if possible
        try:
            from app.memory.redis_client import get_redis
            r = get_redis()
            keys = r.keys("mediai:semantic_cache:*")
            if keys:
                r.delete(*keys)
                logger.info("[SemanticCache] Cleared %d keys from Redis", len(keys))
        except Exception as exc:
            logger.debug("[SemanticCache] Redis clear error: %s", exc)

        with self._lock:
            self._fallback_store.clear()
            self._hits = 0
            self._misses = 0

        return cleared_count

    def stats(self) -> dict[str, Any]:
        """
        Return semantic cache statistics and health.
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        vector_count = 0
        try:
            collection = self._get_collection()
            vector_count = collection.count()
        except Exception:
            with self._lock:
                vector_count = len(self._fallback_store)

        return {
            "enabled": self.settings.enable_semantic_cache,
            "threshold": self.settings.semantic_cache_threshold,
            "ttl_seconds": self.settings.semantic_cache_ttl,
            "total_cached_queries": vector_count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCache:
    """Return a cached singleton SemanticCache instance."""
    return SemanticCache()


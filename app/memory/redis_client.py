"""
Redis session memory client.

Stores per-session conversation state as JSON in Redis with a configurable TTL.
Used by the Memory Agent to load context before the Planner and save context
after the Evidence Synthesis Agent / clinical summary agent.
"""

from __future__ import annotations

import json # Redis stores strings, bytes, numbers. cannot store Python dictionaries directly. So we use json to serialize and deserialize Python dictionaries.
import logging
from functools import lru_cache 
from typing import Any 

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10 # Only last 10 turns are stored in Redis to prevent memory overflow.


@lru_cache(maxsize=1) # Cache Redis client. Only one Redis client is created and reused throughout the application lifecycle.

def get_redis() -> redis.Redis:
    """Return a cached Redis client instance."""
    settings = get_settings()
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    logger.info("Redis client connected to %s", settings.redis_url)
    return client

def _session_key(session_id: str) -> str: # Creates a unique key for each session in Redis. 
    return f"mediai:session:{session_id}" 


def load_session(session_id: str) -> dict[str, Any]:
    """
    Load session data from Redis.

    Returns an empty session dict if no data exists for this session_id or Redis is unreachable.
    """
    empty_state = {
        "conversation_history": [],
        "clinical_topic": "",
        "previous_evidence_refs": [],
    }
    try:
        client = get_redis()
        key = _session_key(session_id)

        raw = client.get(key)
        if not raw:
            return empty_state

        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Redis] Corrupt session data for %s — returning empty.", session_id)
        return empty_state
    except Exception as exc:
        logger.warning("[Redis] Unable to connect to Redis (%s) — using in-memory fallback.", exc)
        return empty_state


def save_session(
    session_id: str,
    query: str,
    summary: str,
    clinical_topic: str,
    evidence_refs: list[str],
) -> None:
    """
    Save updated session data to Redis.

    Appends the current turn to conversation_history (capped at MAX_HISTORY_TURNS).
    """
    try:
        settings = get_settings()
        client = get_redis()
        key = _session_key(session_id)

        session = load_session(session_id)

        # Append current turn
        session["conversation_history"].append(
            {"query": query, "summary": summary}
        )
        # Cap history length
        session["conversation_history"] = session["conversation_history"][-MAX_HISTORY_TURNS:]

        # Update clinical topic if provided
        if clinical_topic:
            session["clinical_topic"] = clinical_topic

        # Merge new evidence refs (deduplicated)
        existing_refs = set(session.get("previous_evidence_refs", []))
        existing_refs.update(evidence_refs)
        session["previous_evidence_refs"] = sorted(existing_refs)

        client.setex(key, settings.redis_session_ttl, json.dumps(session))
        logger.debug("[Redis] Session saved for %s (%d turns)", session_id, len(session["conversation_history"]))
    except Exception as exc:
        logger.warning("[Redis] Unable to save session to Redis (%s). Skipping.", exc)

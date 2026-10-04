"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.schemas import ClinicalSummaryResponse, SafetyResponse

logger = logging.getLogger(__name__)

# Request / Response schemas


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field(default="default", max_length=128) # Used for Multi-turn conversations 


class ChatResponse(BaseModel):
    session_id: str
    is_safe: bool
    safety_response: Optional[SafetyResponse] = None
    response: Optional[ClinicalSummaryResponse] = None
    cached: bool = Field(default=False, description="Whether this response was served from the semantic cache")
    similarity_score: Optional[float] = Field(default=None, description="Cosine similarity score for cached query match")
    matched_query: Optional[str] = Field(default=None, description="Original query that matched the semantic cache")


class CacheStatsResponse(BaseModel):
    enabled: bool
    threshold: float
    ttl_seconds: int
    total_cached_queries: int
    hits: int
    misses: int
    hit_rate: float


class IngestResponse(BaseModel):
    status: str
    indexed: int    # number of new/modified PDFs processed
    skipped: int    # number of unchanged PDFs skipped
    collections: list[str]


class UploadIngestResponse(BaseModel):
    status: str
    filename: str
    collection: str
    chunks_indexed: int
    pages_processed: int
    message: str = "Document uploaded and indexed successfully"



class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    services: dict[str, str]

# Application lifespan — run ingestion on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the app begins serving requests."""
    logger.info("MediAI LangGraph starting up …")
    settings = get_settings()

    # Incremental ingestion on startup (never force-rebuild here)
    try:
        from app.rag.ingest import run_ingestion
        result = await run_ingestion(force=False)
        logger.info(
            "Startup ingestion complete — indexed=%d skipped=%d",
            result["indexed"],
            result["skipped"],
        )
    except Exception as exc:
        logger.warning("Startup ingestion failed (non-fatal): %s", exc)

    # Purge expired semantic cache entries on startup
    try:
        if settings.enable_semantic_cache:
            from app.memory.semantic_cache import get_semantic_cache
            purged = get_semantic_cache().purge_expired()
            if purged > 0:
                logger.info("Startup cache purge complete — removed %d expired entries", purged)
    except Exception as exc:
        logger.debug("Startup cache purge skipped/failed (non-fatal): %s", exc)

    yield

    logger.info("MediAI LangGraph shutting down.")


# FastAPI app

app = FastAPI(
    title="MediAI LangGraph",
    description=(
        "Multi-Agent Clinical Evidence Synthesis Platform. "
        "Retrieves, ranks, and synthesises evidence from clinical guidelines "
        "and biomedical literature. For educational use only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend Static Files 
from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/")


# Routes

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return service status and downstream connectivity."""
    services: dict[str, str] = {}

    # Redis ping
    try:
        from app.memory.redis_client import get_redis
        r = get_redis()
        r.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"

    # Chroma ping
    try:
        from app.rag.vectorstore import get_chroma_client
        client = get_chroma_client()
        client.heartbeat()
        services["chroma"] = "ok"
    except Exception as exc:
        services["chroma"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in services.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        services=services,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Clinical"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the full multi-agent clinical evidence synthesis workflow with semantic caching.
    1. Check Semantic Cache
    2. If found return the cached response. If not found, run the langgraph workflow & Save response to Semantic Cache.     
    """
    settings = get_settings()

    # Semantic Cache Lookup 
    if settings.enable_semantic_cache:
        try:
            from app.memory.semantic_cache import get_semantic_cache

            cache = get_semantic_cache()
            cached_match = cache.lookup(request.query)

            if cached_match is not None:
                # Update session memory so user's multi-turn history stays consistent
                if request.session_id and cached_match.get("is_safe") and cached_match.get("response"):
                    try:
                        from app.memory.redis_client import save_session

                        resp_data = cached_match["response"]
                        summary_text = (
                            resp_data.get("summary", "")
                            if isinstance(resp_data, dict)
                            else getattr(resp_data, "summary", "")
                        )
                        citations = (
                            resp_data.get("citations", [])
                            if isinstance(resp_data, dict)
                            else getattr(resp_data, "citations", [])
                        )
                        evidence_refs = [
                            c.get("reference_id", "")
                            if isinstance(c, dict)
                            else getattr(c, "reference_id", "")
                            for c in citations
                        ]
                        save_session(
                            session_id=request.session_id,
                            query=request.query,
                            summary=summary_text,
                            clinical_topic="",
                            evidence_refs=[r for r in evidence_refs if r],
                        )
                    except Exception as exc:
                        logger.debug("Failed to update session memory on cache hit: %s", exc)

                return ChatResponse(
                    session_id=request.session_id,
                    is_safe=cached_match["is_safe"],
                    safety_response=cached_match.get("safety_response"),
                    response=cached_match.get("response"),
                    cached=True,
                    similarity_score=cached_match.get("similarity"),
                    matched_query=cached_match.get("matched_query"),
                )   
        except Exception as exc:
            logger.warning("Semantic cache lookup encountered error: %s", exc)

    # Full Multi-Agent Graph Execution
    try:
        from app.graph.graph import get_graph

        graph = get_graph()
        initial_state = {
            "query": request.query,
            "session_id": request.session_id,
            "reflection_count": 0,
            "guideline_evidence": [],
            "pubmed_evidence": [],
            "ranked_evidence": [],
            "memory_context": {},
        }

        final_state = await graph.ainvoke(initial_state)

        # Store in Semantic Cache 
        if settings.enable_semantic_cache:
            try:
                from app.memory.semantic_cache import get_semantic_cache

                cache = get_semantic_cache()

                # caches the safety violation response so future identical or semantically similar unsafe queries can be blocked instantly without re-running the graph.
                if not final_state.get("is_safe", True): 
                    cache.store(
                        query=request.query,
                        is_safe=False,
                        safety_response=final_state.get("safety_response"),
                        session_id=request.session_id,
                    )
                elif final_state.get("final_response"): # caches successful response
                    cache.store(
                        query=request.query,
                        is_safe=True,
                        response=final_state.get("final_response"),
                        session_id=request.session_id,
                    )
            except Exception as exc:
                logger.warning("Failed to store in semantic cache: %s", exc)

        # Returning the Final Response 

        if not final_state.get("is_safe", True):
            return ChatResponse(
                session_id=request.session_id,
                is_safe=False,
                safety_response=final_state.get("safety_response"),
                cached=False,
            )

        return ChatResponse(
            session_id=request.session_id,
            is_safe=True,
            response=final_state.get("final_response"),
            cached=False,
        )

    except Exception as exc:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
async def cache_stats() -> CacheStatsResponse:
    """Return semantic cache statistics including hits, misses, and count."""
    try:
        from app.memory.semantic_cache import get_semantic_cache

        stats = get_semantic_cache().stats()
        return CacheStatsResponse(**stats)
    except Exception as exc:
        logger.exception("Failed to retrieve cache stats")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/cache/clear", tags=["Cache"])
async def clear_cache() -> dict[str, Any]:
    """Clear all entries from the semantic cache."""
    try:
        from app.memory.semantic_cache import get_semantic_cache

        cleared = get_semantic_cache().clear()
        return {"status": "ok", "cleared_entries": cleared}
    except Exception as exc:
        logger.exception("Failed to clear cache")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    force: bool = Query(
        default=False,
        description="Set to true to force a full rebuild of all Chroma collections.",
    )
) -> IngestResponse:
    """
    Batch sync or rebuild Chroma collections from the server's knowledge directory.
    - Default (force=false): Incremental sync — only processes new or modified PDFs on disk.
    - force=true: Administrative wipe and rebuild of all Chroma collections from scratch.
    """
    try:
        from app.rag.ingest import run_ingestion

        result = await run_ingestion(force=force)
        return IngestResponse(
            status="ok",
            indexed=result["indexed"],
            skipped=result["skipped"],
            collections=result["collections"],
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/upload", response_model=UploadIngestResponse, tags=["RAG"])
async def upload_and_ingest(
    file: UploadFile = File(
        ...,
        description="PDF guideline or research document to upload and index into Chroma.",
    ),
    collection: str = Form(
        default="clinical_guidelines",
        description="Target collection: clinical_guidelines, consensus_reports, research_articles, future_documents",
    ),
) -> UploadIngestResponse:
    """
    Upload a new PDF file via multipart form-data, save it to the knowledge base,
    and immediately chunk and index it into the selected Chroma collection.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF (.pdf) files are supported.",
        )

    try:
        from app.rag.ingest import ingest_uploaded_pdf

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await ingest_uploaded_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
            target_collection=collection,
        )

        return UploadIngestResponse(
            status=result["status"],
            filename=result["filename"],
            collection=result["collection"],
            chunks_indexed=result["chunks_indexed"],
            pages_processed=result["pages_processed"],
            message=f"Successfully uploaded and indexed {result['chunks_indexed']} chunks into '{result['collection']}'.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to upload and ingest PDF: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.get("/collections", response_model=CollectionsResponse, tags=["RAG"])
async def list_collections() -> CollectionsResponse:
    """List all Chroma collections and their document counts."""
    try:
        from app.rag.vectorstore import get_chroma_client

        client = get_chroma_client()
        collections = client.list_collections()
        infos = [
            CollectionInfo(name=col.name, document_count=col.count())
            for col in collections
        ]
        return CollectionsResponse(collections=infos)
    except Exception as exc:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

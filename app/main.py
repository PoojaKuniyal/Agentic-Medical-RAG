"""
FastAPI application entrypoint.

Routes
  GET  /health                      — Service health check
  POST /chat                        — Main multi-agent clinical query endpoint
  POST /ingest                      — Trigger incremental PDF ingestion
  POST /ingest?force=true           — Force full rebuild of all Chroma collections
  GET  /collections                 — List Chroma collections and document counts
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import ClinicalSummaryResponse, SafetyResponse

logger = logging.getLogger(__name__)

# Request / Response schemas


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field(default="default", max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    is_safe: bool
    safety_response: Optional[SafetyResponse] = None
    response: Optional[ClinicalSummaryResponse] = None


class IngestResponse(BaseModel):
    status: str
    indexed: int    # number of new/modified PDFs processed
    skipped: int    # number of unchanged PDFs skipped
    collections: list[str]


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

# ── Mount Frontend Static Files ──────────────────────────────────────────────
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
    Run the full multi-agent clinical evidence synthesis workflow.

    The query passes through:
    Guardrail → Memory → Planner → [Guideline RAG ‖ PubMed] →
    Evidence Ranking → Clinical Reasoning → Reflection → Clinical Summary →
    Memory update
    """
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

        if not final_state.get("is_safe", True):
            return ChatResponse(
                session_id=request.session_id,
                is_safe=False,
                safety_response=final_state.get("safety_response"),
            )

        return ChatResponse(
            session_id=request.session_id,
            is_safe=True,
            response=final_state.get("final_response"),
        )

    except Exception as exc:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    force: bool = Query(
        default=False,
        description="Set to true to force a full rebuild of all Chroma collections.",
    )
) -> IngestResponse:
    """
    Trigger PDF ingestion from the knowledge directory.

    - Default: incremental — only processes new or modified PDFs.
    - force=true: clears and rebuilds all Chroma collections from scratch.
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

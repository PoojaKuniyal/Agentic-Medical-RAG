"""
PDF ingestion pipeline for Chroma.

Steps:
1. Map subdirectories to Chroma collections
2. Load and compare manifest hashes
3. Detect new or modified PDFs
4. Chunk and embed changed PDFs into collections
5. Update manifest with latest hashes

force=True -> rebuilds all collections from scratch

New PDF      -> insert new chunks
Modified PDF -> delete old chunks + insert new chunks
Unchanged    -> skip
"""

from __future__ import annotations
 
import logging
from pathlib import Path

from app.config import get_settings
from app.rag.chunker import chunk_pages
from app.rag.loader import load_pdf
from app.rag.manifest import (
    compute_hash,
    find_changed_pdfs,
    load_manifest,
    save_manifest,
)
from app.rag.vectorstore import (
    delete_chunks_by_source_pdf,
    delete_collection,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


async def run_ingestion(force: bool = False) -> dict:
    """
    Run the incremental (or forced) ingestion pipeline.

    Parameters
    ----------
    force : If True, delete all Chroma collections and rebuild from scratch.

    Returns
    -------
    dict with keys: indexed (int), skipped (int), collections (list[str])
    """
    settings = get_settings()
    knowledge_dir = Path(settings.knowledge_dir).resolve()
    collection_map: dict[str, str] = settings.chroma_collections

    if not knowledge_dir.exists():
        logger.warning("Knowledge directory does not exist: %s", knowledge_dir)
        return {"indexed": 0, "skipped": 0, "collections": []}

    manifest = load_manifest(knowledge_dir)

    if force:
        logger.info("Force rebuild requested — clearing all Chroma collections.")
        for collection_name in collection_map.values():
            delete_collection(collection_name)
        manifest = {}

    new_or_modified, unchanged = find_changed_pdfs(knowledge_dir, manifest)

    indexed = 0
    collections_touched: set[str] = set()

    for pdf_path in new_or_modified:
        relative_key = pdf_path.relative_to(knowledge_dir).as_posix()
        is_modified = relative_key in manifest

        # Determine which Chroma collection this PDF belongs to
        subdir = pdf_path.parent.name
        collection_name = collection_map.get(subdir, "future_documents")

        logger.info(
            "Ingesting: %s (modified=%s) → collection=%s",
            pdf_path.name,
            is_modified,
            collection_name,
        )

        try:
            if is_modified:
                delete_chunks_by_source_pdf(collection_name, pdf_path.name)

            pages = load_pdf(pdf_path, knowledge_dir)
            chunks = chunk_pages(pages)
            count = upsert_chunks(collection_name, chunks)
            indexed += count

            # Update manifest with the new hash only upon successful ingestion
            manifest[relative_key] = compute_hash(pdf_path)
            collections_touched.add(collection_name)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", pdf_path.name, exc)
            # Continue processing remaining PDFs — do not abort entire run

    save_manifest(knowledge_dir, manifest)

    logger.info(
        "Ingestion complete — indexed=%d chunks from %d PDFs, skipped=%d PDFs",
        indexed,
        len(new_or_modified),
        len(unchanged),
    )

    return {
        "indexed": len(new_or_modified),
        "skipped": len(unchanged),
        "collections": sorted(collections_touched),
    }

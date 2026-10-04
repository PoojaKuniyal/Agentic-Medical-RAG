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
    force : If True, delete all Chroma collections and rebuild from scratch.
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
            "Ingesting: %s (modified=%s) -> collection=%s",
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


async def ingest_uploaded_pdf(
    file_bytes: bytes,
    filename: str,
    target_collection: str = "clinical_guidelines",
) -> dict:
    """
    Save an uploaded PDF into the corresponding knowledge directory subfolder
    and ingest its chunks into the specified Chroma collection.
    """
    settings = get_settings()
    knowledge_dir = Path(settings.knowledge_dir).resolve()

    valid_collections = list(settings.chroma_collections.values())
    if target_collection not in valid_collections:
        target_collection = "future_documents"

    # Destination directory: knowledge_dir / target_collection
    target_dir = knowledge_dir / target_collection
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name
    if not safe_filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    dest_path = target_dir / safe_filename
    dest_path.write_bytes(file_bytes)

    manifest = load_manifest(knowledge_dir)
    relative_key = dest_path.relative_to(knowledge_dir).as_posix()
    is_modified = relative_key in manifest

    if is_modified:
        delete_chunks_by_source_pdf(target_collection, dest_path.name)

    pages = load_pdf(dest_path, knowledge_dir)
    chunks = chunk_pages(pages)
    count = upsert_chunks(target_collection, chunks)

    manifest[relative_key] = compute_hash(dest_path)
    save_manifest(knowledge_dir, manifest)

    logger.info(
        "Uploaded and ingested '%s' into '%s' (%d chunks, %d pages)",
        safe_filename,
        target_collection,
        count,
        len(pages),
    )

    return {
        "filename": safe_filename,
        "collection": target_collection,
        "chunks_indexed": count,
        "pages_processed": len(pages),
        "status": "ok",
    }


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(level=logging.INFO)
    is_force = "--force" in sys.argv
    logger.info("Starting ingestion script (force=%s)...", is_force)
    result = asyncio.run(run_ingestion(force=is_force))
    print(f"\n--- INGESTION COMPLETE ---")
    print(f"Indexed Chunks : {result.get('indexed', 0)}")
    print(f"Skipped PDFs   : {result.get('skipped', 0)}")
    print(f"Collections    : {result.get('collections', [])}\n")



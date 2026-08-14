"""
Text chunking using LangChain's RecursiveCharacterTextSplitter.

This file takes each page and breaks it into smaller overlapping pieces/ chunks.

Chunk size and overlap are configurable via .env (CHUNK_SIZE, CHUNK_OVERLAP).
Each chunk retains all metadata from its source page.
"""

from __future__ import annotations

import logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as err:
    raise ImportError(
        "langchain-text-splitters package is required for document chunking. "
        "Please install it using 'pip install langchain-text-splitters'."
    ) from err

from app.config import get_settings

logger = logging.getLogger(__name__)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split a list of page records into smaller text chunks.

    Parameters
    ----------
    pages : list of page dicts (from loader.load_pdf)
        Each dict must have: text, source_pdf, collection, page_number.

    Returns
    -------
    list of chunk dicts with keys:
        text, source_pdf, collection, page_number, chunk_index
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk_text,
                    "source_pdf": page["source_pdf"],
                    "collection": page["collection"],
                    "page_number": page["page_number"],
                    "chunk_index": idx,
                }
            )

    logger.debug(
        "Chunked %d pages into %d chunks (size=%d overlap=%d)",
        len(pages),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks

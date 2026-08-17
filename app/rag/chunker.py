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
    Split a list of page records into continuous text chunks across page boundaries
    based on semantic text continuity, while preserving starting page_number metadata.

    Parameters
    ----------
    pages : list of page dicts (from loader.load_pdf)
        Each dict must have: text, source_pdf, collection, page_number.

    Returns
    -------
    list of chunk dicts with keys:
        text, source_pdf, collection, page_number, chunk_index
    """
    if not pages:
        return []

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    # 1. Concatenate all page texts into a continuous document string
    # and record character offset intervals for each page
    full_text_parts: list[str] = []
    page_offsets: list[tuple[int, int, int]] = []  # (start_char, end_char, page_number)
    current_offset = 0

    for page in pages:
        text = page["text"]
        start_char = current_offset
        full_text_parts.append(text)
        current_offset += len(text)
        end_char = current_offset
        page_offsets.append((start_char, end_char, page["page_number"]))

        # Add page joiner delimiter and account for its length in character offsets
        full_text_parts.append("\n\n")
        current_offset += 2

    full_text = "".join(full_text_parts)
    source_pdf = pages[0]["source_pdf"]
    collection = pages[0]["collection"]

    # 2. Continuous semantic splitting — creates Document objects with metadata["start_index"]
    docs = splitter.create_documents([full_text])

    chunks: list[dict] = []

    # 3. Assign starting page_number metadata using exact start_index character offset lookup
    for idx, doc in enumerate(docs):
        chunk_text = doc.page_content
        start_char = doc.metadata.get("start_index", 0)

        # Lookup starting page number based on character offset
        start_page = pages[0]["page_number"]
        for p_start, p_end, p_num in page_offsets:
            if p_start <= start_char < p_end:
                start_page = p_num
                break

        chunks.append(
            {
                "text": chunk_text,
                "source_pdf": source_pdf,
                "collection": collection,
                "page_number": start_page,
                "chunk_index": idx,
            }
        )

    logger.debug(
        "Chunked %d pages into %d continuous semantic chunks for %s (size=%d, overlap=%d)",
        len(pages),
        len(chunks),
        source_pdf,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks

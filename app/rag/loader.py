"""
PDF loader using PyMuPDF (fitz).

Take a PDF -> extract its text page by page -> attach metadata -> return structured Python dictionaries

Returns a list of raw page dictionaries, each carrying:
  - page_number  : 1-indexed
  - text         : extracted page text
  - source_pdf   : filename of the source document
  - collection   : Chroma collection derived from the knowledge sub-directory
"""

from __future__ import annotations

import logging
from pathlib import Path

import re

logger = logging.getLogger(__name__)

# Compile regex patterns for boilerplate identification and removal
_BOILERPLATE_LINE_PATTERNS = [
    # Standalone page counters: "Page 36 of 131", "Page 36", "PAGE 1"
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    # Standalone copyright / legal lines: "© NICE 2026. All rights reserved.", "Copyright © 2026"
    re.compile(r"^\s*(?:©|\(c\)|copyright)\s+.*$", re.IGNORECASE),
    re.compile(r"^\s*all\s+rights\s+reserved\.?\s*$", re.IGNORECASE),
    # Standalone terms & conditions URLs / legal footers
    re.compile(r"^\s*https?://\S*(?:terms|legal|conditions|privacy)\S*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:www\.)?\S+\.\S+/(?:terms|legal|conditions|privacy)\S*\s*$", re.IGNORECASE),
]

_BOILERPLATE_INLINE_PATTERNS = [
    # Inline page counters: "Page 36 of 131"
    (re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE), ""),
    # Inline copyright notices: "© NICE 2026. All rights reserved." or "© NICE 2026"
    (re.compile(r"©\s*NICE\s*\d{4}\.?\s*(?:All\s+rights\s+reserved\.?)?", re.IGNORECASE), ""),
    (re.compile(r"(?:©|\(c\)|copyright)\s+\d{4}\b.*?(?:all\s+rights\s+reserved\.?)?", re.IGNORECASE), ""),
    # Terms URLs embedded in text lines
    (re.compile(r"https?://\S*terms\S*", re.IGNORECASE), ""),
    (re.compile(r"www\.\S+/(?:terms|legal|conditions|privacy)\S*", re.IGNORECASE), ""),
]


def clean_page_text(text: str) -> str:
    """
    Clean extracted raw page text by stripping boilerplate header/footer noise:
      - "Page X of Y" or standalone "Page X"
      - Copyright / legal notices (e.g. "© NICE 2026. All rights reserved.")
      - Repeated terms & conditions URLs

    Clinical headers, section titles, definitions, recommendations, tables,
    and narrative text are preserved intact.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Check standalone boilerplate line patterns
        is_boilerplate = any(pattern.match(stripped) for pattern in _BOILERPLATE_LINE_PATTERNS)
        if is_boilerplate:
            continue

        # Apply inline boilerplate removal
        line_clean = line
        for pattern, replacement in _BOILERPLATE_INLINE_PATTERNS:
            line_clean = pattern.sub(replacement, line_clean)

        # If line still has meaningful text after inline stripping, keep it
        if line_clean.strip():
            cleaned_lines.append(line_clean.strip())

    return "\n".join(cleaned_lines).strip()


def _collection_from_path(pdf_path: Path, knowledge_dir: Path) -> str:
    """
    Derive the Chroma collection name from the PDF's parent subdirectory.

    Look at where the PDF is located and determine which Chroma collection it belongs to.

    Example:
        knowledge/clinical_guidelines/...pdf  →  "clinical_guidelines"
        knowledge/consensus_reports/....pdf     →  "consensus_reports"
    """
    try:
        relative = pdf_path.relative_to(knowledge_dir)
        collection = relative.parts[0]
        if collection in ["clinical_guidelines", "consensus_reports", "future_documents", "research_articles"]:
            return collection
        return "future_documents"
    except (ValueError, IndexError):
        return "future_documents"


def load_pdf(pdf_path: Path, knowledge_dir: Path) -> list[dict]:
    """
    Load a PDF file and return a list of cleaned page-level records. 

    Parameters
    ----------
    pdf_path : Absolute path to the PDF file.
    knowledge_dir : Root knowledge directory (used to derive the collection name).

    Returns
    -------
    list of page dicts with keys: page_number, text, source_pdf, collection. One dictionary represents one page.
    Empty pages (or pages with only stripped boilerplate) are skipped.
    Physical 1-indexed page numbers are preserved as metadata.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF loading. "
            "Install it with: pip install pymupdf"
        ) from e

    collection = _collection_from_path(pdf_path, knowledge_dir)
    pages: list[dict] = []
    modified_pages_count = 0

    logger.info("Loading PDF: %s (collection=%s)", pdf_path.name, collection)

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        for page_index in range(total_pages):
            page = doc[page_index] # Get the page
            raw_text = page.get_text().strip() # extracts text from that page, and strips any leading/trailing whitespace. 

            if not raw_text:
                continue  # skip blank pages

            cleaned_text = clean_page_text(raw_text)

            if not cleaned_text:
                modified_pages_count += 1
                continue  # skip pages that contained only boilerplate

            if cleaned_text != raw_text:
                modified_pages_count += 1

            # Appending the page to our list. Building Page-level records Metadata + text   
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                    "source_pdf": pdf_path.name,
                    "collection": collection,
                }
            )

    logger.debug(
        "Boilerplate cleaned from %d / %d pages in %s",
        modified_pages_count,
        total_pages,
        pdf_path.name,
    )
    logger.info("  → %d non-empty pages loaded from %s", len(pages), pdf_path.name)
    return pages


"""
SHA-256 hash-based manifest for incremental PDF ingestion.

The manifest is stored at knowledge/.manifest.json and tracks the last-seen
hash of every PDF file. Files are only re-ingested when their hash changes.
This ensures the Chroma vector database is never rebuilt unnecessarily.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.json"


def _manifest_path(knowledge_dir: Path) -> Path:
    return knowledge_dir / MANIFEST_FILENAME


def load_manifest(knowledge_dir: Path) -> dict[str, str]:
    """
    Load the existing manifest.

    Returns
    -------
    dict mapping relative PDF path (str) → SHA-256 hex digest (str).
    Returns an empty dict if no manifest exists yet.
    """
    path = _manifest_path(knowledge_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read manifest at %s: %s — starting fresh.", path, exc)
        return {}


def save_manifest(knowledge_dir: Path, manifest: dict[str, str]) -> None:
    """Persist the updated manifest to disk."""
    path = _manifest_path(knowledge_dir)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.debug("Manifest saved to %s (%d entries)", path, len(manifest))


def compute_hash(pdf_path: Path) -> str:
    """Return the SHA-256 hex digest of a PDF file."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_changed_pdfs(
    knowledge_dir: Path,
    manifest: dict[str, str],
) -> tuple[list[Path], list[Path]]:
    """
    Scan knowledge_dir recursively for PDFs and identify changes.

    Parameters
    ----------
    knowledge_dir : Root of the knowledge directory (contains collection subdirectories).
    manifest : The currently persisted manifest.

    Returns
    -------
    new_or_modified : PDFs that are new or whose hash differs from the manifest.
    unchanged : PDFs whose hash matches the manifest — no action needed.
    """
    new_or_modified: list[Path] = []
    unchanged: list[Path] = []

    for pdf_path in sorted(knowledge_dir.rglob("*.pdf")):
        relative_key = pdf_path.relative_to(knowledge_dir).as_posix()
        current_hash = compute_hash(pdf_path)

        if manifest.get(relative_key) != current_hash:
            logger.info("Detected new/modified PDF: %s", relative_key)
            new_or_modified.append(pdf_path)
        else:
            unchanged.append(pdf_path)

    return new_or_modified, unchanged

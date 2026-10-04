"""
PubMed search tool implementation using NCBI Entrez REST API.

Uses Biopython's Entrez module to Search PubMed & fetch article details.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.graph.schemas import PubMedArticle

logger = logging.getLogger(__name__)


class EntrezPubMedTool:
    """PubMed search via NCBI Entrez REST API (Biopython)."""

    def __init__(self) -> None:
        try:
            from Bio import Entrez
        except ImportError as e:
            raise ImportError(
                "biopython is required for EntrezPubMedTool. "
                "Install it with: pip install biopython"
            ) from e

        settings = get_settings()
        Entrez.email = settings.pubmed_email or "mediai@example.com"
        self._max_results = settings.pubmed_max_results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def search(self, query: str, max_results: Optional[int] = None) -> list[PubMedArticle]:
        """
        Search PubMed and return structured article records.
        """
        from Bio import Entrez

        n = max_results or self._max_results
        logger.info("PubMed search: query='%s' max=%d", query, n)

        # Step 1: Search — get PMIDs
        with Entrez.esearch(db="pubmed", term=query, retmax=n) as handle:
            search_results = Entrez.read(handle)

        pmids: list[str] = search_results.get("IdList", [])
        if not pmids:
            logger.info("PubMed search returned 0 results for query: %s", query)
            return []

        logger.info("PubMed returned %d PMIDs, fetching details …", len(pmids))

        # Step 2: Fetch article details
        with Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        ) as handle:
            xml_data = handle.read()

        articles = _parse_pubmed_xml(xml_data)
        logger.info("Parsed %d articles from PubMed XML", len(articles))
        return articles


def get_pubmed_tool() -> EntrezPubMedTool:
    """Factory function that returns the EntrezPubMedTool instance."""
    return EntrezPubMedTool()


# XML parsing 


def _parse_pubmed_xml(xml_data: bytes) -> list[PubMedArticle]:
    """Parse PubMed efetch XML and return a list of PubMedArticle objects."""
    root = ET.fromstring(xml_data)
    articles: list[PubMedArticle] = []

    for article_node in root.findall(".//PubmedArticle"):
        try:
            articles.append(_parse_single_article(article_node))
        except Exception as exc:
            logger.warning("Failed to parse article node: %s", exc)
            continue

    return articles


def _parse_single_article(node: ET.Element) -> PubMedArticle:
    """Extract fields from a single <PubmedArticle> XML node."""

    def text(xpath: str, default: str = "") -> str:
        el = node.find(xpath)
        return el.text.strip() if el is not None and el.text else default

    # PMID
    pmid = text(".//PMID")

    # Title
    title = text(".//ArticleTitle")

    # Authors
    authors: list[str] = []
    for author in node.findall(".//Author"):
        last = text_from(author, "LastName")
        fore = text_from(author, "ForeName")
        if last:
            authors.append(f"{last} {fore}".strip())

    # Journal
    journal = text(".//Journal/Title") or text(".//MedlineTA")

    # Year
    year_str = (
        text(".//PubDate/Year")
        or text(".//PubDate/MedlineDate")[:4]
    )
    year: Optional[int] = int(year_str) if year_str.isdigit() else None

    # DOI
    doi: Optional[str] = None
    for id_node in node.findall(".//ArticleId"):
        if id_node.get("IdType") == "doi":
            doi = id_node.text.strip() if id_node.text else None
            break

    # Abstract
    abstract_parts = [
        el.text.strip()
        for el in node.findall(".//AbstractText")
        if el.text
    ]
    abstract = " ".join(abstract_parts)

    # Publication types
    pub_types = [
        el.text.strip()
        for el in node.findall(".//PublicationType")
        if el.text
    ]

    return PubMedArticle(
        pmid=pmid,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        doi=doi,
        abstract=abstract,
        publication_types=pub_types,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    )


def text_from(node: ET.Element, tag: str, default: str = "") -> str:
    """Helper: get text from a child element of a given node."""
    el = node.find(tag)
    return el.text.strip() if el is not None and el.text else default

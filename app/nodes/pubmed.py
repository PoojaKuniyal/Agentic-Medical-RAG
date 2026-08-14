"""
PubMed — biomedical literature retrieval only.

Calls the AbstractPubMedTool (Entrez by default, MCP-swappable).
Extracts concise medical search terms from user queries before searching PubMed.
Populates pubmed_evidence, pubmed_query, and pubmed_error_reason in state.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ClinicalState, PubMedArticle
from app.llm.factory import get_llm
from app.tools.base import get_pubmed_tool

logger = logging.getLogger(__name__)

PUBMED_SEARCH_EXTRACTION_PROMPT = (
    "You are a medical literature search expert. "
    "Given a clinical user query, extract 2 to 4 core biomedical search terms or MeSH concepts "
    "suitable for querying PubMed / NCBI Entrez.\n\n"
    "STRICT RULES:\n"
    "1. Remove comparison instructions (e.g., 'compare', 'versus', 'differ', 'comparison'), conversational language, filler words, "
    "and references to PubMed or literature ('PubMed', 'evidence', 'literature', 'studies').\n"
    "2. Exclude guideline organizations, standard bodies, or publishers (e.g., 'NICE', 'ADA', 'EASD', 'CDC', 'WHO', 'guidelines', 'recommendations').\n"
    "3. Preserve the core clinical concepts, disease names, drug classes, and treatment intent (e.g., 'Type 2 Diabetes Mellitus first-line treatment').\n"
    "4. Output ONLY the clean medical concepts separated by spaces on a single line, with no punctuation, quotes, markdown, or extra prose."
)


def _regex_fallback_cleanup(query: str) -> str:
    """Fallback deterministic regex keyword extraction if LLM query generation fails."""
    cleaned = re.sub(
        r"(?i)\b(compare|versus|comparison|according to|nice|ada|easd|cdc|who|guidelines?|recommendations?|pubmed|evidence|literature|studies|what|is|the|are|for|in|of|and|with|patient|adults?)\b",
        " ",
        query,
    )
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else query


def _extract_medical_keywords(query: str) -> str:
    """
    Extract concise, clean medical search terms from user query using the LLM as the primary mechanism,
    falling back to deterministic regex cleanup if the LLM is unavailable or fails.
    """
    try:
        llm = get_llm(temperature=0.0)
        messages = [
            SystemMessage(content=PUBMED_SEARCH_EXTRACTION_PROMPT),
            HumanMessage(content=f"User Query: {query}"),
        ]
        response = llm.invoke(messages) # acts as a query-translation agent step to formulate optimal biomedical search terms 
        extracted = response.content.strip().replace("\n", " ").strip("\"'")
        # Post-process to ensure no stray comparison, guideline, or filler tokens leaked through
        extracted = re.sub(
            r"(?i)\b(NICE|ADA|EASD|CDC|WHO|guidelines?|recommendations?|pubmed|evidence|literature|studies|compare|versus|comparison)\b",
            " ",
            extracted,
        )
        extracted = " ".join(extracted.split())
        if extracted and len(extracted) >= 3:
            logger.info("[PubMedAgent] Extracted search terms via LLM: '%s'", extracted)
            return extracted
    except Exception as exc:
        logger.warning("[PubMedAgent] LLM query extraction failed (%s) — using regex fallback.", exc)

    cleaned_fallback = _regex_fallback_cleanup(query)
    logger.info("[PubMedAgent] Extracted search terms via regex fallback: '%s'", cleaned_fallback)
    return cleaned_fallback


def _filter_relevant_articles(raw_query: str, articles: list[PubMedArticle]) -> list[PubMedArticle]:
    """Filter PubMed articles to ensure they contain core clinical keywords from the query."""
    if not articles:
        return []

    # Extract non-filler medical keywords from the query for matching
    keywords = [
        w.lower() for w in re.split(r"\W+", raw_query)
        if len(w) > 3 and w.lower() not in {
            "according", "nice", "guideline", "guidelines", "recommendation", "recommendations",
            "what", "where", "when", "which", "with", "from", "that", "this", "have", "does"
        }
    ]

    if not keywords:
        return articles

    relevant_articles = []
    for article in articles:
        text_content = f"{article.title} {article.abstract}".lower()
        # Keep article if it mentions at least one significant query keyword
        if any(kw in text_content for kw in keywords):
            relevant_articles.append(article)
        else:
            logger.info("[PubMedAgent] Filtered out irrelevant PubMed article PMID %s: '%s'", article.pmid, article.title[:80])

    return relevant_articles


def run_pubmed(state: ClinicalState) -> dict:
    """
    LangGraph node function for the PubMed Agent.
    Hybrid LangGraph Node that performs both an LLM task and an external tool execution 

    Reads:  query
    Writes: pubmed_evidence, pubmed_query, pubmed_error_reason, pubmed_results_count, filtered_pubmed_results_count
    """
    raw_query = state.get("query", "")
    logger.info("[PubMedAgent] Processing user query: '%s'", raw_query[:80])

    medical_query = _extract_medical_keywords(raw_query) # LLM task
    tool = get_pubmed_tool() # external tool execution

    raw_articles: list[PubMedArticle] = []
    filtered_articles: list[PubMedArticle] = []
    error_reason: str | None = None

    try:
        raw_articles = tool.search(medical_query)
        logger.info("[PubMedAgent] Retrieved %d articles from PubMed", len(raw_articles))
        filtered_articles = _filter_relevant_articles(raw_query, raw_articles)
        logger.info("[PubMedAgent] Retained %d relevant articles after filtering", len(filtered_articles))
        if not filtered_articles:
            error_reason = f"No relevant PubMed articles retained for search query: '{medical_query}'"
    except Exception as exc:
        error_reason = f"PubMed API error: {exc}"
        logger.error("[PubMedAgent] PubMed search failed: %s", exc)
        raw_articles = []
        filtered_articles = []

    return {
        "pubmed_evidence": filtered_articles,
        "pubmed_query": medical_query,
        "pubmed_error_reason": error_reason,
        "pubmed_results_count": len(raw_articles),
        "filtered_pubmed_results_count": len(filtered_articles),
    }


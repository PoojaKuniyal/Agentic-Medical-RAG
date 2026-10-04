"""
Evidence Synthesis Agent.

Produces the final structured response:
  - Summary
  - Evidence Support (objective retrieval metrics)
  - Evidence Sources
  - Citations
  - Follow-up Questions

NEVER diagnoses patients. NEVER provides emergency medical advice.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.schemas import (
    Citation,
    ClinicalSummaryResponse,
    EvidenceSource,
    EvidenceSupport,
    GuidelineChunk,
    PubMedArticle,
    RankedEvidence,
)
from app.graph.state import ClinicalState
from app.llm.factory import get_llm
from app.prompts import CLINICAL_SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _format_lightweight_metadata(ranked_evidence: list[RankedEvidence]) -> str:
    """Prepare lightweight evidence metadata (excluding raw text chunks) for the LLM prompt."""
    items = []
    for item in ranked_evidence[:10]:  # cap at top 10 for prompt length
        items.append(
            {
                "rank": item.rank,
                "evidence_type": item.evidence_type,
                "source_type": item.source_type,
                "title": item.title,
                "source_ref": item.source_ref,
                "collection": item.collection,
            }
        )
    return json.dumps(items, indent=2)


def _build_programmatic_citations_and_sources(
    ranked_evidence: list[RankedEvidence],
    pubmed_evidence: list[PubMedArticle],
    guideline_evidence: list[GuidelineChunk],
) -> tuple[list[Citation], list[EvidenceSource]]:
    """Construct exact Citation and EvidenceSource lists from ranked evidence state."""
    pubmed_map = {p.pmid: p for p in pubmed_evidence if p.pmid}
    
    citations: list[Citation] = []
    evidence_sources: list[EvidenceSource] = []
    seen_citation_keys = set()
    seen_source_keys = set()
    citation_counter = 1

    for item in ranked_evidence[:10]:
        src_type = "clinical_guideline" if item.source_type == "guideline" else item.source_type
        
        # Deduplicate evidence sources
        src_key = (src_type, item.title)
        if src_key not in seen_source_keys:
            seen_source_keys.add(src_key)
            evidence_sources.append(
                EvidenceSource(
                    source_type=src_type,
                    collection=item.collection,
                    title=item.title,
                    evidence_type=item.evidence_type,
                )
            )
        
        if item.source_type == "pubmed":
            cite_key = ("pubmed", item.source_ref or item.title)
            if cite_key in seen_citation_keys:
                continue
            seen_citation_keys.add(cite_key)

            ref_id = f"REF-{citation_counter:03d}"
            citation_counter += 1

            article = pubmed_map.get(item.source_ref)
            pmid_val = item.source_ref if (item.source_ref and item.source_ref.isdigit()) else None
            journal_val = article.journal if article and article.journal else None
            year_val = article.year if article and article.year else None
            doi_val = article.doi if article and article.doi else None
            authors_val = article.authors if article and article.authors else None
            
            citations.append(
                Citation(
                    reference_id=ref_id,
                    source_type="pubmed",
                    title=item.title,
                    authors=authors_val,
                    year=year_val,
                    journal=journal_val,
                    doi=doi_val,
                    pmid=pmid_val,
                )
            )
        else:
            page_val = None
            for g in guideline_evidence:
                if g.source_pdf == item.source_ref and g.page is not None:
                    page_val = g.page
                    break
            
            cite_key = ("guideline", item.source_ref or item.title, page_val)
            if cite_key in seen_citation_keys:
                continue
            seen_citation_keys.add(cite_key)

            ref_id = f"REF-{citation_counter:03d}"
            citation_counter += 1

            citations.append(
                Citation(
                    reference_id=ref_id,
                    source_type="clinical_guideline",
                    title=item.title,
                    page=page_val,
                )
            )

    return citations, evidence_sources


def run_clinical_summary(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Evidence Synthesis Agent.

    Reads:  query, reasoning, ranked_evidence, evidence_support, pubmed_evidence, guideline_evidence
    Writes: final_response
    """
    query = state.get("query", "")
    reasoning = state.get("reasoning", "")
    ranked_evidence = state.get("ranked_evidence", [])
    pubmed_evidence = state.get("pubmed_evidence", [])
    guideline_evidence = state.get("guideline_evidence", [])
    evidence_support = state.get("evidence_support", EvidenceSupport())

    logger.info("[ClinicalSummaryAgent] Generating final structured response …")

    metadata_str = _format_lightweight_metadata(ranked_evidence)
    prog_citations, prog_sources = _build_programmatic_citations_and_sources(
        ranked_evidence, pubmed_evidence, guideline_evidence
    )

    llm = get_llm(temperature=0.1)

    support_summary = (
        evidence_support.retrieval_relevance_summary if evidence_support else "Retrieved evidence items loaded."
    )

    messages = [
        SystemMessage(content=CLINICAL_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Clinical question:\n{query}\n\n"
                f"Clinical reasoning:\n{reasoning}\n\n"
                f"Evidence Support Summary: {support_summary}\n\n"
                f"Retrieved Sources Metadata:\n{metadata_str}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw_content = response.content
    if isinstance(raw_content, list):
        raw_content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content)
    raw = str(raw_content).strip()
    import re
    # 1. Remove <think> reasoning blocks
    raw_cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    # 2. Extract JSON object {...}
    json_match = re.search(r"\{.*\}", raw_cleaned, re.DOTALL)
    json_str = json_match.group(0) if json_match else raw_cleaned

    try:
        data = json.loads(json_str, strict=False)
        final_response = _build_response(data, evidence_support, prog_citations, prog_sources)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "[ClinicalSummaryAgent] Could not parse LLM response — building fallback. Error: %s",
            exc,
        )
        final_response = _build_fallback_response(
            reasoning, ranked_evidence, evidence_support
        )

    logger.info("[ClinicalSummaryAgent] Final response generated.")
    return {"final_response": final_response}


def _clean_summary_text(text: str) -> str:
    """Clean summary text by stripping <think> blocks, inline REF tags, and internal debug metadata strings."""
    if not text:
        return ""
    import re
    # 1. Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2. Remove standalone <think> or </think> tags
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("**", "").replace("__", "")
    # 3. Remove inline [REF-xxx] tags
    cleaned = re.sub(r"\[REF-\d+\]", "", cleaned, flags=re.IGNORECASE)
    # 4. Remove debug metadata strings
    cleaned = re.sub(r"\s*\(\s*(source|rank|PMID|PDF reference|collection):.*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?i)\b(PMID:\s*unavailable|PDF reference:\s*[\w.-]+|source:\s*rank\s*\d+)\b", "", cleaned)
    # 5. Remove "Here's a thinking process:" lines
    cleaned = re.sub(r"(?i)^.*here'?s a thinking process.*$", "", cleaned, flags=re.MULTILINE)
    
    # Clean multiple spaces and return
    lines = [re.sub(r"\s+", " ", l).strip() for l in cleaned.split("\n")]
    return "\n".join([l for l in lines if l])


def _sanitize_citation(cite_dict: dict) -> dict:
    """Sanitize citation dict to ensure unavailable fields are None and page numbers are integers."""
    res = dict(cite_dict)
    for k in ["pmid", "doi", "journal", "title"]:
        val = res.get(k)
        if isinstance(val, str) and (val.strip().lower() in ("unavailable", "none", "n/a", "null", "")):
            res[k] = None

    raw_page = res.get("page")
    if raw_page is not None:
        if isinstance(raw_page, int):
            pass
        elif isinstance(raw_page, str):
            import re
            m = re.search(r"\d+", raw_page)
            res["page"] = int(m.group(0)) if m else None
        else:
            res["page"] = None
    return res


def _build_response(
    data: dict,
    evidence_support: EvidenceSupport,
    prog_citations: list[Citation],
    prog_sources: list[EvidenceSource],
) -> ClinicalSummaryResponse:
    """Build a ClinicalSummaryResponse from parsed LLM JSON output and programmatic metadata."""

    evidence_sources = (
        [EvidenceSource(**src) for src in data.get("evidence_sources", [])]
        if data.get("evidence_sources")
        else prog_sources
    )
    citations = (
        [Citation(**_sanitize_citation(cite)) for cite in data.get("citations", [])]
        if data.get("citations")
        else prog_citations
    )

    summary_text = _clean_summary_text(data.get("summary", ""))

    return ClinicalSummaryResponse(
        summary=summary_text,
        evidence_support=evidence_support,
        evidence_sources=evidence_sources,
        citations=citations,
        follow_up_questions=data.get("follow_up_questions", []),
    )


def _build_fallback_response(
    reasoning: str,
    ranked_evidence: list[RankedEvidence],
    evidence_support: EvidenceSupport,
) -> ClinicalSummaryResponse:
    """Minimal fallback response when LLM JSON parsing fails."""
    citations = []
    seen_refs = set()
    counter = 1
    for item in ranked_evidence[:10]:
        key = (item.source_type, item.source_ref or item.title)
        if key in seen_refs:
            continue
        seen_refs.add(key)
        ref_id = f"REF-{counter:03d}"
        counter += 1
        if item.source_type == "pubmed":
            citations.append(
                Citation(
                    reference_id=ref_id,
                    source_type="pubmed",
                    title=item.title,
                    pmid=item.source_ref if item.source_ref and item.source_ref.isdigit() else None,
                )
            )
        else:
            citations.append(
                Citation(
                    reference_id=ref_id,
                    source_type="clinical_guideline",
                    title=item.title,
                )
            )

    seen_sources = set()
    evidence_sources = []
    for item in ranked_evidence[:10]:
        key = (item.source_type, item.title)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        evidence_sources.append(
            EvidenceSource(
                source_type=item.source_type,
                collection=item.collection,
                title=item.title,
                evidence_type=item.evidence_type,
            )
        )

    cleaned_reasoning = _clean_summary_text(reasoning)
    lines = [line.strip() for line in cleaned_reasoning.split("\n") if line.strip() and not line.strip().startswith("#")]
    formatted_points = []
    for idx, line in enumerate(lines, 1):
        if line[0].isdigit() and (line[1:3] in (". ", ") ")):
            formatted_points.append(line)
        else:
            formatted_points.append(f"{idx}. {line}")

    summary_fallback = "\n".join(formatted_points) if formatted_points else "1. Retrieved evidence items were analyzed."

    return ClinicalSummaryResponse(
        summary=summary_fallback,
        evidence_support=evidence_support,
        evidence_sources=evidence_sources,
        citations=citations,
        follow_up_questions=[
            "What additional clinical trial data are available for this comparison?",
            "How do national guidelines address these findings?"
        ],
    )



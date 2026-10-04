"""
Reflection Agent - quality-control loop

Reviews the reasoning output and decides whether to continue or trigger another
retrieval cycle. Hard cap: maximum MAX_REFLECTION_ITERATIONS iterations.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.graph.state import ClinicalState
from app.llm.factory import get_llm
from app.prompts import REFLECTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def run_reflection(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Reflection Agent.

    Reads:  reasoning, reflection_count, ranked_evidence, evidence_support
    Writes: reflection_needed, reflection_count
    """
    settings = get_settings()
    max_iterations = settings.max_reflection_iterations

    reasoning = state.get("reasoning", "")
    reflection_count = state.get("reflection_count", 0)
    ranked_evidence = state.get("ranked_evidence", [])

    logger.info(
        "[ReflectionAgent] Reviewing reasoning (iteration %d/%d, evidence items=%d) …",
        reflection_count + 1,
        max_iterations,
        len(ranked_evidence),
    )

    # Hard cap — always continue after max iterations
    if reflection_count >= max_iterations:
        logger.info(
            "[ReflectionAgent] Max iterations reached (%d) — forcing CONTINUE.",
            max_iterations,
        )
        return {"reflection_needed": False, "reflection_count": reflection_count}

    llm = get_llm(model_tier="fast", temperature=0.0)

    messages = [
        SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"reflection_count: {reflection_count}\n"
                f"max_iterations: {max_iterations}\n"
                f"evidence_item_count: {len(ranked_evidence)}\n\n"
                f"Reasoning to review:\n{reasoning}"
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
        decision = data.get("decision", "CONTINUE").upper()
        reason = data.get("reason", "")
        issues = data.get("issues_detected", [])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[ReflectionAgent] Parse error — defaulting to CONTINUE. Error: %s", exc)
        decision = "CONTINUE"
        reason = "Parse error — proceeding to summary."
        issues = []

    if decision == "RETRY_RETRIEVAL":
        logger.info(
            "[ReflectionAgent] RETRY requested — reason: %s | issues: %s",
            reason,
            issues,
        )
        return {
            "reflection_needed": True,
            "reflection_count": reflection_count + 1,
        }

    logger.info("[ReflectionAgent] CONTINUE — reason: %s", reason)
    return {
        "reflection_needed": False,
        "reflection_count": reflection_count,
    }


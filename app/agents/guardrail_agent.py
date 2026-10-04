"""
Guardrail Agent (first graph node) — safety gate.
Its job is to decide whether the user's query is allowed to proceed into the clinical evidence workflow. 
It uses an LLM with structured output to analyze the query and determine if it is safe to proceed.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.schemas import GuardrailDecision, SafetyResponse
from app.graph.state import ClinicalState
from app.llm.factory import get_llm
from app.prompts import GUARDRAIL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def run_guardrail(state: ClinicalState) -> dict:
    """
    LangGraph node function for the Guardrail Agent.
    """
    query = state["query"]
    logger.info("Guardrail Agent - Evaluating query safety …")

    try:
        llm = get_llm(model_tier="fast", temperature=0.0)
        structured_llm = llm.with_structured_output(GuardrailDecision, method="json_mode")

        messages = [
            SystemMessage(
                content=GUARDRAIL_SYSTEM_PROMPT
                + "\n\nRespond strictly with JSON matching this schema:\n"
                '{"is_safe": bool, "reason": str|null, "message": str|null}'
            ),
            HumanMessage(content=f"Query to evaluate:\n\n{query}"),
        ]

        decision = structured_llm.invoke(messages)

        if isinstance(decision, dict):
            decision = GuardrailDecision(**decision)
        elif not isinstance(decision, GuardrailDecision):
            raise ValueError(f"Unexpected decision output type: {type(decision)}")

        if not decision.is_safe:
            safety_response = SafetyResponse(
                blocked=True,
                reason=decision.reason or "unknown",
                message=decision.message or "This query cannot be processed.",
            )
            logger.warning(
                "[GuardrailAgent] Query BLOCKED — reason=%s", safety_response.reason
            )
            return {"is_safe": False, "safety_response": safety_response}

        logger.info("[GuardrailAgent] Query SAFE — continuing.")
        return {"is_safe": True, "safety_response": None}

    except Exception as exc:
        logger.error(
            "[GuardrailAgent] Safety evaluation failed or model output invalid: %s — Failing CLOSED for safety.",
            exc,
            exc_info=True,
        )
        fail_closed_response = SafetyResponse(
            blocked=True,
            reason="guardrail_error",
            message="This query cannot be processed due to a temporary safety check failure.",
        )
        return {"is_safe": False, "safety_response": fail_closed_response}

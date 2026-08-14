"""
Prompt loader module for MediAI.

Dynamically loads Markdown (.md) system prompts from the prompts directory.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt markdown file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


CLINICAL_REASONING_SYSTEM_PROMPT = _load_prompt("CLINICAL_REASONING_SYSTEM_PROMPT.md")
CLINICAL_SUMMARY_SYSTEM_PROMPT = _load_prompt("CLINICAL_SUMMARY_SYSTEM_PROMPT.md")
GUARDRAIL_SYSTEM_PROMPT = _load_prompt("GUARDRAIL_SYSTEM_PROMPT.md")
PLANNER_SYSTEM_PROMPT = _load_prompt("PLANNER_SYSTEM_PROMPT.md")
REFLECTION_SYSTEM_PROMPT = _load_prompt("REFLECTION_SYSTEM_PROMPT.md")

__all__ = [
    "CLINICAL_REASONING_SYSTEM_PROMPT",
    "CLINICAL_SUMMARY_SYSTEM_PROMPT",
    "GUARDRAIL_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "REFLECTION_SYSTEM_PROMPT",
]

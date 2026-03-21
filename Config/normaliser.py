"""
src/utils/normaliser.py
=======================
Map free-form LLM output → canonical relation label.

Strategy (in order):
  1. Already canonical → return as-is
  2. None / empty      → return NO_INFO
  3. Substring match   → handles verbose outputs ("the answer is activation")
  4. GPT-3.5 fallback  → summarise unknown terms to canonical label
"""

import string
from Configs.config import RELATIONS, NO_INFO


def _clean(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).strip().lower()


def normalize_relation(value: str | None) -> str:
    if value is None:
        return NO_INFO

    cleaned = _clean(value)
    if cleaned in RELATIONS or cleaned == NO_INFO:
        return cleaned

    # Substring match
    for rel in RELATIONS:
        if rel in cleaned:
            return rel

    # GPT-3.5 fallback summarisation — lazy import to avoid circular deps
    try:
        from src.models.gpt_backend    import GPTBackend
        from Configs.model_config      import get_model_config
        backend  = GPTBackend(get_model_config("gpt-3.5"))
        question = (
            f"Summarize the term '{value}' to exactly one of: "
            "activation, inhibition, phosphorylation, or no information. "
            "Reply with only that single term."
        )
        _, result = backend.query("", question)
        r = _clean(result) if result else NO_INFO
        return r if r in RELATIONS else NO_INFO
    except Exception:
        return NO_INFO

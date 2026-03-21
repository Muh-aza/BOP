"""
src/models/cohere_backend.py
============================
Cohere Command-R+ backend.

Model details:
  - 52B parameter model optimised for enterprise applications
  - 4096-token context, custom tokenizer
  - Python SDK v4.2.1 with automatic retry and connection pooling
"""

import string
import time

import cohere

from Configs.config import COHERE_API_KEY, MAX_RETRIES
from Configs.model_config import ModelConfig


class CohereBackend:
    """Query Cohere Command-R+ with exponential-backoff retry."""

    def __init__(self, cfg: ModelConfig):
        self.cfg    = cfg
        self.client = cohere.Client(COHERE_API_KEY)

    def query(self, system_prompt: str, user_question: str) -> tuple[str | None, str | None]:
        """
        Send a chat request to Cohere.
        system_prompt maps to the 'preamble' parameter.

        Returns
        -------
        (raw, cleaned) — both None on permanent failure.
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat(
                    model       = self.cfg.model_id,
                    preamble    = system_prompt,
                    message     = user_question,
                    temperature = self.cfg.temperature,
                    p           = self.cfg.top_p,
                    max_tokens  = self.cfg.max_tokens,
                )
                raw     = response.text.strip().lower()
                cleaned = raw.translate(
                    str.maketrans("", "", string.punctuation)
                ).strip()
                return raw, cleaned

            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(2 ** attempt)

        return None, None

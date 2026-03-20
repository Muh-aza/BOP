"""
src/models/gpt_backend.py
=========================
OpenAI backend for GPT-3.5 / GPT-4 / GPT-4o.

Paper Methods 4.1:
  - Max 5 retries with exponential backoff (delays 1-60 s)
  - Temperature = 0.7, top_p = 0.9, max_tokens = 512
  - Frequency penalty = 0.0, presence penalty = 0.0
"""

import string
import time

import openai

from Configs.config import (
    OPENAI_API_KEY, TEMPERATURE, TOP_P,
    MAX_OUTPUT_TOKENS, MAX_RETRIES,
)
from Configs.model_config import ModelConfig


class GPTBackend:
    """Query any OpenAI chat model with exponential-backoff retry."""

    def __init__(self, cfg: ModelConfig):
        self.cfg       = cfg
        openai.api_key = OPENAI_API_KEY

    def query(self, system_prompt: str, user_question: str) -> tuple[str | None, str | None]:
        """
        Send a chat-completion request.

        Returns
        -------
        (raw, cleaned) — raw is original text, cleaned has punctuation stripped.
        Both None on permanent failure after MAX_RETRIES attempts.
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model             = self.cfg.model_id,
                    messages          = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_question},
                    ],
                    max_tokens        = self.cfg.max_tokens,
                    temperature       = self.cfg.temperature,
                    top_p             = self.cfg.top_p,
                    frequency_penalty = 0.0,
                    presence_penalty  = 0.0,
                )
                raw     = response.choices[0].message.content.strip().lower()
                cleaned = raw.translate(
                    str.maketrans("", "", string.punctuation)
                ).strip()
                return raw, cleaned

            except openai.OpenAIError:
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(min(60, 2 ** attempt))   # jittered backoff, max 60 s

            except Exception:
                return None, None

        return None, None

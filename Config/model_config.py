"""
Configs/model_config.py
=======================
Per-model settings for all five LLMs evaluated in BOP-SAP.

Models:
  GPT-3.5  — 175B params, BPE tokenizer, 4097-token context
  GPT-4    — MoE architecture, 8192-token context
  GPT-4o   — Multimodal unified transformer
  Cohere   — Command-R+, 52B params, 4096-token context
  LLaMA-3.1— 8B params, 32 layers, 4096 hidden dim, RoPE, fp16
"""

from dataclasses import dataclass
from Configs.config import (
    TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS,
    LLAMA_MODEL_ID, COHERE_MODEL_ID,
    LLAMA_MAX_NEW_TOKENS, LLAMA_HALF_PRECISION,
)


@dataclass
class ModelConfig:
    model_id        : str
    backend         : str           # "openai" | "cohere" | "huggingface"
    display_name    : str
    max_tokens      : int   = MAX_OUTPUT_TOKENS
    temperature     : float = TEMPERATURE
    top_p           : float = TOP_P
    max_new_tokens  : int   = LLAMA_MAX_NEW_TOKENS   # HuggingFace only
    do_sample       : bool  = False                  # HuggingFace only
    half_precision  : bool  = LLAMA_HALF_PRECISION   # HuggingFace only


MODEL_REGISTRY : dict[str, ModelConfig] = {

    "gpt-3.5": ModelConfig(
        model_id     = "gpt-3.5-turbo",
        backend      = "openai",
        display_name = "GPT-3.5",
    ),
    "gpt-4": ModelConfig(
        model_id     = "gpt-4",
        backend      = "openai",
        display_name = "GPT-4",
    ),
    "gpt-4o": ModelConfig(
        model_id     = "gpt-4o",
        backend      = "openai",
        display_name = "GPT-4o",
    ),
    "cohere": ModelConfig(
        model_id     = COHERE_MODEL_ID,
        backend      = "cohere",
        display_name = "Cohere Command-R+",
    ),
    "llama3": ModelConfig(
        model_id       = LLAMA_MODEL_ID,
        backend        = "huggingface",
        display_name   = "LLaMA-3.1-8B",
        max_new_tokens = LLAMA_MAX_NEW_TOKENS,
        do_sample      = False,
        half_precision = LLAMA_HALF_PRECISION,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Retrieve a ModelConfig by name.
    Accepted: 'gpt-3.5', 'gpt-4', 'gpt-4o', 'cohere', 'llama3'.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]

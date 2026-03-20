"""
src/models/model_factory.py
============================
Factory — instantiates the correct backend for a given model name.
Keeps LLaMA loaded as a singleton to avoid reloading between calls.
"""

from Configs.model_config import get_model_config
from src.models.gpt_backend    import GPTBackend
from src.models.cohere_backend import CohereBackend
from src.models.llama_backend  import LlamaBackend

# Singleton cache — LLaMA-3.1 takes ~30 s to load; load only once
_llama_instance: LlamaBackend | None = None


def build_backend(model_name: str):
    """
    Return the correct backend instance for *model_name*.

    Parameters
    ----------
    model_name : 'gpt-3.5' | 'gpt-4' | 'gpt-4o' | 'cohere' | 'llama3'

    Returns
    -------
    GPTBackend | CohereBackend | LlamaBackend
    """
    global _llama_instance
    cfg = get_model_config(model_name)

    if cfg.backend == "openai":
        return GPTBackend(cfg)

    if cfg.backend == "cohere":
        return CohereBackend(cfg)

    if cfg.backend == "huggingface":
        if _llama_instance is None:
            _llama_instance = LlamaBackend(cfg)
        return _llama_instance

    raise ValueError(f"Unknown backend '{cfg.backend}' for model '{model_name}'")

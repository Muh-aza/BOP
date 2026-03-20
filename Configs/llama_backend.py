"""
src/models/llama_backend.py
===========================
LLaMA-3.1-8B-Instruct local inference backend.

Paper Methods 4.1:
  - 8B params, 32 transformer layers, 4096 hidden dim
  - RoPE positional embeddings, 8192-token context
  - fp16 mixed precision, 12 GB VRAM utilisation
  - Deployed via HuggingFace Transformers + CUDA 11.8

Two capabilities:
  1. Gene-relation text generation (predict())
  2. Layer-wise hidden-state extraction (build_layer_embeddings())
     → Used for CHI, cosine similarity, UMAP (Fig 3 / S2-S4)
"""

import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from Configs.config import HF_TOKEN, LLAMA_MODEL_ID, LLAMA_MAX_NEW_TOKENS
from Configs.model_config import ModelConfig

_RELATION_RE = re.compile(r"(activation|inhibition|phosphorylation)", re.I)


class LlamaBackend:
    """LLaMA-3.1-8B-Instruct: prediction + hidden-state extraction."""

    def __init__(self, cfg: ModelConfig | None = None):
        device_id   = 0 if torch.cuda.is_available() else -1
        self.device = "cuda" if device_id == 0 else "cpu"

        print(f"Loading {LLAMA_MODEL_ID} on {self.device} (fp16) ...")

        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(LLAMA_MODEL_ID, token=HF_TOKEN)
            .half()        # fp16 — 12 GB VRAM (paper Methods 4.1)
            .to(self.device)
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model     = self.model,
            tokenizer = self.tokenizer,
            device    = device_id,
        )

    # ── Gene-relation prediction ───────────────────────────────────────────────
    def query(self, system_prompt: str, user_question: str) -> tuple[str, str]:
        """
        Run text generation and return (raw, canonical_label).
        canonical_label ∈ {activation, inhibition, phosphorylation, no information}.
        """
        full_prompt = f"{system_prompt}\nQ: {user_question}"
        try:
            out   = self.pipe(full_prompt, max_new_tokens=LLAMA_MAX_NEW_TOKENS,
                              do_sample=False)
            text  = out[0]["generated_text"]
            match = _RELATION_RE.search(text)
            if match:
                label = match.group(1).lower()
                return label, label
            return "no information", "no information"
        except Exception:
            return "no information", "no information"

    # ── Hidden-state extraction ────────────────────────────────────────────────
    def get_hidden_states(self, texts: list[str]) -> list[np.ndarray]:
        """
        Extract mean-pooled hidden states for all transformer layers.

        Returns
        -------
        List of length num_layers, each ndarray shape (len(texts), hidden_dim).
        """
        enc = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc, output_hidden_states=True)

        return [h.mean(dim=1).cpu().float().numpy() for h in outputs.hidden_states]

    def build_layer_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Build (num_layers, num_texts, hidden_dim) array.
        Processes one text at a time to avoid GPU OOM.
        """
        all_rows = [
            self.get_hidden_states([text])
            for text in tqdm(texts, desc="LLaMA-3.1 embeddings")
        ]
        num_layers = len(all_rows[0])
        num_texts  = len(texts)
        hidden_dim = all_rows[0][0].shape[1]

        emb = np.zeros((num_layers, num_texts, hidden_dim), dtype=np.float32)
        for row_idx, layers in enumerate(all_rows):
            for layer_idx, vec in enumerate(layers):
                emb[layer_idx, row_idx] = vec[0]
        return emb

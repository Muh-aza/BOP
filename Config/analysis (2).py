"""
src/models/llama_backend.py
===========================
LLaMA-3.1-8B-Instruct local inference backend.

Model details:
  - 8B params, 32 transformer layers + 1 embedding layer = 33 total layers
  - Hidden dimension: 4096
  - RoPE positional embeddings, 8192-token context
  - fp16 mixed precision, ~12 GB VRAM
  - Deployed via HuggingFace Transformers + CUDA

Embedding extraction:
  - All 33 hidden states extracted (layer 0 = embedding, layers 1-32 = transformer)
  - Three pooling strategies: mean, last-token, first-token
  - Output shape: (33, num_texts, 4096)
  - Batch processing with configurable batch size to avoid GPU OOM
"""

import re
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from Configs.config import HF_TOKEN, LLAMA_MODEL_ID, LLAMA_MAX_NEW_TOKENS
from Configs.model_config import ModelConfig

_RELATION_RE = re.compile(r"(activation|inhibition|phosphorylation)", re.I)

PoolingMode = Literal["mean", "last_token", "first_token"]


class LlamaBackend:
    """
    LLaMA-3.1-8B-Instruct: gene-relation prediction + full all-layer
    hidden-state extraction for embedding analysis.

    LLaMA-3.1-8B has 33 hidden states:
      layer 0  — token embedding layer (before any transformer block)
      layers 1-32 — transformer decoder blocks
    All 33 layers are extracted and returned for downstream CHI,
    cosine similarity, and UMAP analyses.
    """

    NUM_LAYERS = 33   # 1 embedding + 32 transformer blocks
    HIDDEN_DIM = 4096

    def __init__(
        self,
        cfg:          ModelConfig | None = None,
        batch_size:   int                = 8,
        pooling_mode: PoolingMode        = "mean",
    ):
        """
        Parameters
        ----------
        cfg          : ModelConfig (optional — uses defaults from config.py)
        batch_size   : Number of texts processed per GPU forward pass.
                       Reduce to 1-4 if you see CUDA OOM errors.
        pooling_mode : How to pool token-level hidden states into one vector.
                       'mean'        — average over all non-padding tokens (default)
                       'last_token'  — use the last non-padding token only
                       'first_token' — use the [BOS] token (position 0)
        """
        self.batch_size   = batch_size
        self.pooling_mode = pooling_mode

        device_id   = 0 if torch.cuda.is_available() else -1
        self.device = "cuda" if device_id == 0 else "cpu"

        print(f"Loading {LLAMA_MODEL_ID} on {self.device} (fp16) ...")
        print(f"  Pooling mode : {pooling_mode}")
        print(f"  Batch size   : {batch_size}")
        print(f"  Total layers : {self.NUM_LAYERS}  (embedding + 32 transformer blocks)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_ID, token=HF_TOKEN,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(LLAMA_MODEL_ID, token=HF_TOKEN)
            .half()
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
        Run text generation and return (raw_output, canonical_label).

        canonical_label ∈ {activation, inhibition, phosphorylation, no information}
        """
        full_prompt = f"{system_prompt}\nQ: {user_question}"
        try:
            out   = self.pipe(
                full_prompt,
                max_new_tokens = LLAMA_MAX_NEW_TOKENS,
                do_sample      = False,
            )
            text  = out[0]["generated_text"]
            match = _RELATION_RE.search(text)
            if match:
                label = match.group(1).lower()
                return label, label
            return "no information", "no information"
        except Exception:
            return "no information", "no information"

    # ── Hidden-state extraction — all 33 layers ────────────────────────────────

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """
        Pool token-level hidden states → one vector per sample.

        Parameters
        ----------
        hidden         : (batch, seq_len, hidden_dim)  float16 tensor on GPU
        attention_mask : (batch, seq_len)               1=real token, 0=padding

        Returns
        -------
        (batch, hidden_dim) float32 numpy array
        """
        if self.pooling_mode == "mean":
            mask      = attention_mask.unsqueeze(-1).float()
            summed    = (hidden.float() * mask).sum(dim=1)
            counts    = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).cpu().numpy()

        elif self.pooling_mode == "last_token":
            lengths = attention_mask.sum(dim=1) - 1          # index of last real token
            idx     = lengths.clamp(min=0).long()
            batch_i = torch.arange(hidden.size(0), device=hidden.device)
            return hidden.float()[batch_i, idx].cpu().numpy()

        else:  # first_token — BOS at position 0
            return hidden.float()[:, 0, :].cpu().numpy()

    def _forward_batch(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """
        Run one forward pass on a batch of texts.

        Returns
        -------
        List of 33 arrays, each shape (len(texts), hidden_dim).
        Layers are ordered: [embedding, block_1, ..., block_32].
        """
        enc = self.tokenizer(
            texts,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
            max_length      = 512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **enc,
                output_hidden_states = True,
            )

        pooled = [
            self._pool(hs, enc["attention_mask"])
            for hs in outputs.hidden_states
        ]
        return pooled

    def get_hidden_states(self, texts: list[str]) -> list[np.ndarray]:
        """
        Extract pooled hidden states for all 33 layers for a list of texts.

        Processes in batches of self.batch_size to avoid GPU OOM.

        Parameters
        ----------
        texts : List of input strings

        Returns
        -------
        List of 33 arrays, each shape (len(texts), 4096).
        Index 0 = embedding layer, index 1-32 = transformer blocks.
        """
        all_batches: list[list[np.ndarray]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_batches.append(self._forward_batch(batch))

        num_layers = len(all_batches[0])
        result: list[np.ndarray] = [
            np.concatenate([b[layer] for b in all_batches], axis=0)
            for layer in range(num_layers)
        ]
        return result

    def build_layer_embeddings(
        self,
        texts:    list[str],
        desc:     str = "LLaMA-3.1 embeddings",
    ) -> np.ndarray:
        """
        Build the full (num_layers, num_texts, hidden_dim) embedding array
        covering all 33 layers of LLaMA-3.1-8B.

        Parameters
        ----------
        texts : List of input strings
        desc  : tqdm progress bar label

        Returns
        -------
        np.ndarray of shape (33, len(texts), 4096), dtype float32.
          axis 0 — layer index (0 = embedding, 1-32 = transformer blocks)
          axis 1 — sample index
          axis 2 — hidden dimension (4096)
        """
        num_texts  = len(texts)
        all_layers: list[list[np.ndarray]] = []

        for i in tqdm(range(0, num_texts, self.batch_size), desc=desc):
            batch = texts[i : i + self.batch_size]
            all_layers.append(self._forward_batch(batch))

        num_layers = len(all_layers[0])
        hidden_dim = all_layers[0][0].shape[1]

        emb = np.zeros((num_layers, num_texts, hidden_dim), dtype=np.float32)
        offset = 0
        for batch_layers in all_layers:
            b_size = batch_layers[0].shape[0]
            for layer_idx, vec in enumerate(batch_layers):
                emb[layer_idx, offset : offset + b_size] = vec
            offset += b_size

        print(f"  Embeddings shape: {emb.shape}  "
              f"[{num_layers} layers × {num_texts} samples × {hidden_dim} dim]")
        return emb

    def get_layer_names(self) -> list[str]:
        """
        Return human-readable names for all 33 layers.

        Returns
        -------
        ['embedding', 'block_01', 'block_02', ..., 'block_32']
        """
        return ["embedding"] + [f"block_{i:02d}" for i in range(1, self.NUM_LAYERS)]

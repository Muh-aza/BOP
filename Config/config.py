"""
Configs/config.py
=================
Master configuration for BOP-SAP.
All hyperparameters are centrally defined here.
Set API credentials via environment variables — never hardcode secrets.
"""

import os

# ── API Credentials ────────────────────────────────────────────────────────────
OPENAI_API_KEY  : str  = os.getenv("OPENAI_API_KEY",  "")
COHERE_API_KEY  : str  = os.getenv("COHERE_API_KEY",  "")
HF_TOKEN        : str  = os.getenv("HF_TOKEN",         "")

# ── Relation Labels ────────────────────────────────────────────────────────────
RELATIONS : list[str] = ["activation", "inhibition", "phosphorylation"]
NO_INFO   : str       = "no information"

# ── Standardised Inference Hyperparameters ─────────────────────────────────────
TEMPERATURE       : float = 0.7
TOP_P             : float = 0.9
FREQUENCY_PENALTY : float = 0.0
PRESENCE_PENALTY  : float = 0.0
MAX_OUTPUT_TOKENS : int   = 512
MAX_RETRIES       : int   = 5      # exponential backoff, max 5 attempts

# ── Model Identifiers ──────────────────────────────────────────────────────────
LLAMA_MODEL_ID  : str  = "meta-llama/Llama-3.1-8B-Instruct"
COHERE_MODEL_ID : str  = "command-r-plus"
LLAMA_MAX_NEW_TOKENS  : int  = 4
LLAMA_HALF_PRECISION  : bool = True    # fp16 — 12 GB VRAM

# ── Optuna / Bayesian Optimisation ────────────────────────────────────────────
OPTUNA_N_TRIALS         : int = 50
OPTUNA_N_STARTUP_TRIALS : int = 10    # random trials before TPE activates
OPTUNA_DIRECTION        : str = "maximize"

# ── ASCII Structural Anchor ───────────────────────────────────────────────────
ASCII_KEY_LENGTH : int = 15           # 15-char random string per trial

# ── Prompt Component Sheet → Column mapping ───────────────────────────────────
PROMPT_SHEETS : dict[str, str] = {
    "roles":                "role",
    "tasks":                "task",
    "general_instructions": "instruction",
    "user_questions":       "question",
}

# ── Few-Shot Demo block (appended to every system prompt) ─────────────────────
FEW_SHOT_DEMO : str = (
    "Example: "
    "Q: What effect does gene EGF have on gene EGFR? A: Activation. "
    "Q: What effect does gene GRK2 have on gene OR2AJ1? A: Inhibition. "
    "Q: What effect does gene CDK9 have on gene NELFB? A: Phosphorylation. "
    "Answers must be one of activation, inhibition, phosphorylation or no information."
)

# ── UMAP Hyperparameters ──────────────────────────────────────────────────────
UMAP_N_NEIGHBORS  : int   = 30
UMAP_MIN_DIST     : float = 0.1
UMAP_METRIC       : str   = "cosine"
UMAP_RANDOM_STATE : int   = 42
UMAP_GRID_COLS    : int   = 5
PLOT_DPI          : int   = 600

# ── Plot Styling ───────────────────────────────────────────────────────────────
MARKERS : dict[str, str] = {
    "Activation": "o", "Inhibition": "s",
    "Phosphorylation": "^", "Incorrect": "x",
}
COLORS : dict[str, str] = {
    "Activation": "blue", "Inhibition": "red",
    "Phosphorylation": "black", "Incorrect": "purple",
}

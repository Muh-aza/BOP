"""
Configs/__init__.py
====================
Public API — import anything from the Configs package directly.

Usage
-----
from Configs import get_model_config, BASELINE_SAP, TRAINING_CSV, ensure_dirs
from Configs import RELATIONS, NO_INFO, MARKERS, COLORS
"""

from Configs.config import (
    OPENAI_API_KEY, COHERE_API_KEY, HF_TOKEN,
    RELATIONS, NO_INFO,
    TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS, MAX_RETRIES,
    LLAMA_MODEL_ID, LLAMA_MAX_NEW_TOKENS, LLAMA_HALF_PRECISION,
    OPTUNA_N_TRIALS, OPTUNA_N_STARTUP_TRIALS, ASCII_KEY_LENGTH,
    PROMPT_SHEETS, FEW_SHOT_DEMO,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_METRIC,
    UMAP_RANDOM_STATE, UMAP_GRID_COLS, PLOT_DPI,
    MARKERS, COLORS,
)
from Configs.model_config import get_model_config, MODEL_REGISTRY, ModelConfig
from Configs.prompt_config import (
    BASELINE_SAP, SAPTemplate, assemble_system_prompt,
)
from Configs.paths_config import (
    ROOT_DIR, DATA_DIR, RESULTS_DIR, MODEL_DIR,
    PROMPT_PARTS_XLS, TRAINING_CSV, VALIDATION_CSV, TESTING_XLS,
    OPT_RESULTS_XLS, EVAL_RESULTS_XLS,
    CH_VALUES_CSV, COSINE_SIM_CSV, TRAJECTORY_XLS, CONVERGENCE_PNG,
    UMAP_ANCHOR_PNG, UMAP_BASELINE_PNG, UMAP_COMPARE_PNG,
    COSINE_CLASS_PNG, COSINE_AVG_PNG, COSINE_TRAJ_PNG,
    ensure_dirs,
    get_prompt_parts_xls, get_training_csv,
    get_validation_csv, get_testing_xls,
)

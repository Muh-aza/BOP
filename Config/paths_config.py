"""
Configs/paths_config.py
=======================
All filesystem paths for BOP-SAP.
One place to change paths — reflected everywhere.
Override any path via environment variables.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# ── Input Data ─────────────────────────────────────────────────────────────────
DATA_DIR         = ROOT_DIR / "Data" / "dataset" / "Dataset"
PROMPT_PARTS_XLS = DATA_DIR / "prompt_parts.xlsx"
TRAINING_CSV     = DATA_DIR / "training.csv"
VALIDATION_CSV   = DATA_DIR / "validation.csv"
TESTING_XLS      = DATA_DIR / "testing.xlsx"

# ── Outputs ────────────────────────────────────────────────────────────────────
RESULTS_DIR       = ROOT_DIR / "Result"
MODEL_DIR         = ROOT_DIR / "Model"

OPT_RESULTS_XLS   = RESULTS_DIR / "optimization_results.xlsx"
EVAL_RESULTS_XLS  = RESULTS_DIR / "evaluation_results.xlsx"
CH_VALUES_CSV     = RESULTS_DIR / "CH_values.csv"
COSINE_SIM_CSV    = RESULTS_DIR / "cosine_similarity.csv"
TRAJECTORY_XLS    = RESULTS_DIR / "cosine_trajectory.xlsx"
CONVERGENCE_PNG   = RESULTS_DIR / "convergence_curve.png"
UMAP_ANCHOR_PNG   = RESULTS_DIR / "umap_grid_anchor.png"
UMAP_BASELINE_PNG = RESULTS_DIR / "umap_grid_baseline.png"
UMAP_COMPARE_PNG  = RESULTS_DIR / "umap_comparison.png"
COSINE_CLASS_PNG  = RESULTS_DIR / "cosine_classwise.png"
COSINE_AVG_PNG    = RESULTS_DIR / "cosine_average.png"
COSINE_TRAJ_PNG   = RESULTS_DIR / "cosine_trajectory.png"


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in [RESULTS_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Environment-variable overrides ────────────────────────────────────────────
def get_prompt_parts_xls() -> Path:
    return Path(os.getenv("BOP_PROMPT_PARTS", str(PROMPT_PARTS_XLS)))

def get_training_csv() -> Path:
    return Path(os.getenv("BOP_TRAINING_CSV", str(TRAINING_CSV)))

def get_validation_csv() -> Path:
    return Path(os.getenv("BOP_VALID_CSV", str(VALIDATION_CSV)))

def get_testing_xls() -> Path:
    return Path(os.getenv("BOP_TESTING_XLS", str(TESTING_XLS)))

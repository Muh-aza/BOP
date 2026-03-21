"""
src/utils/metrics.py
====================
All evaluation metrics used in BOP-SAP:
  - Precision
  - Recall
  - F1-score per class
  - Macro-F1  — primary Optuna fitness signal
  - MCC       (Matthews Correlation Coefficient)

CHI and Cosine Similarity are computed in their respective analysis modules.
"""

import pandas as pd
from sklearn.metrics import (
    f1_score, matthews_corrcoef,
    precision_score, recall_score,
)

from Configs.config import RELATIONS
from src.utils.normaliser import normalize_relation


def compute_all_metrics(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute the full metrics suite.

    Parameters
    ----------
    df : DataFrame with columns:
         'relation'         — ground-truth label
         'predict_relation' — raw model output string

    Returns
    -------
    dict with keys:
      activation_f1 / inhibition_f1 / phosphorylation_f1
      <rel>_precision  /  <rel>_recall
      macro_f1  /  micro_f1  /  mcc
    """
    df = df.copy()
    df["pred_norm"] = df["predict_relation"].apply(normalize_relation)

    scores: dict[str, float] = {}

    # Per-class F1, Precision, Recall
    for rel in RELATIONS:
        yt = (df["relation"]  == rel).astype(int)
        yp = (df["pred_norm"] == rel).astype(int)
        scores[f"{rel}_f1"]        = f1_score(yt, yp, zero_division=0)
        scores[f"{rel}_precision"] = precision_score(yt, yp, zero_division=0)
        scores[f"{rel}_recall"]    = recall_score(yt, yp, zero_division=0)

    # Macro-F1 — primary Optuna fitness signal
    scores["macro_f1"] = f1_score(
        df["relation"], df["pred_norm"],
        average="macro", zero_division=0, labels=RELATIONS,
    )

    # Micro-F1
    scores["micro_f1"] = f1_score(
        df["relation"], df["pred_norm"],
        average="micro", zero_division=0, labels=RELATIONS,
    )

    # MCC — Matthews Correlation Coefficient
    try:
        scores["mcc"] = matthews_corrcoef(df["relation"], df["pred_norm"])
    except Exception:
        scores["mcc"] = 0.0

    return scores

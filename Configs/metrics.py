"""
src/utils/metrics.py
====================
All evaluation metrics reported in the paper (Equations 9-14):
  - Precision           (Eq. 9)
  - Recall              (Eq. 10)
  - F1-score per class  (Eq. 11)
  - Macro-F1            — primary Optuna fitness signal
  - CHI                 (Eq. 12) — computed in embedding_analysis.py
  - Cosine Similarity   (Eq. 13) — computed in cosine_analysis.py
  - MCC                 (Eq. 14, Fig 2f)
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
    Compute the full metrics suite reported in the paper.

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

    # Per-class F1, Precision, Recall (paper Eq. 9-11, Fig 4)
    for rel in RELATIONS:
        yt = (df["relation"]  == rel).astype(int)
        yp = (df["pred_norm"] == rel).astype(int)
        scores[f"{rel}_f1"]        = f1_score(yt, yp, zero_division=0)
        scores[f"{rel}_precision"] = precision_score(yt, yp, zero_division=0)
        scores[f"{rel}_recall"]    = recall_score(yt, yp, zero_division=0)

    # Macro-F1 — primary Optuna fitness signal (paper Eq. 7 / Fig 2)
    scores["macro_f1"] = f1_score(
        df["relation"], df["pred_norm"],
        average="macro", zero_division=0, labels=RELATIONS,
    )

    # Micro-F1
    scores["micro_f1"] = f1_score(
        df["relation"], df["pred_norm"],
        average="micro", zero_division=0, labels=RELATIONS,
    )

    # MCC — paper Eq. 14, Fig 2f
    try:
        scores["mcc"] = matthews_corrcoef(df["relation"], df["pred_norm"])
    except Exception:
        scores["mcc"] = 0.0

    return scores

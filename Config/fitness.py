"""
src/optimization/fitness.py
============================
Evaluate one candidate SAP configuration on a labelled CSV.
Returns macro-F1 — the Optuna fitness signal.
"""

import pandas as pd

from Configs.config import NO_INFO
from Configs.prompt_config import assemble_system_prompt
from src.models.model_factory import build_backend
from src.utils.metrics import compute_all_metrics


def evaluate_prompt(
    model_name:  str,
    role:        str,
    aims:        str,
    description: str,
    question:    str,
    ascii_key:   str,
    data_csv:    str,
) -> float:
    """
    Assemble the SAP, query the model for every gene pair in data_csv,
    and return macro-averaged F1.

    Parameters
    ----------
    model_name  : 'gpt-3.5' | 'gpt-4' | 'gpt-4o' | 'cohere' | 'llama3'
    role        : SAP Role component
    aims        : SAP Aims component
    description : SAP Description component
    question    : User-turn template — must contain {gene1} and {gene2}
    ascii_key   : 15-char ASCII structural anchor
    data_csv    : CSV with columns: starter, receiver, relation_name

    Returns
    -------
    Macro-F1 (float in [0, 1])
    """
    df      = pd.read_csv(data_csv)
    backend = build_backend(model_name)
    system  = assemble_system_prompt(role, aims, description, ascii_key)

    records = []
    for _, row in df.sample(frac=1).iterrows():
        gene1    = str(row["starter"]).upper()
        gene2    = str(row["receiver"]).upper()
        true_rel = row["relation_name"]
        user_q   = question.format(gene1=gene1, gene2=gene2)
        _, pred  = backend.query(system, user_q)
        records.append({
            "relation":         true_rel,
            "predict_relation": pred if pred is not None else NO_INFO,
        })

    metrics = compute_all_metrics(pd.DataFrame(records))
    return metrics["macro_f1"]

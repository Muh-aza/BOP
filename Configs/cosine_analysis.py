"""
src/analysis/cosine_analysis.py
================================
Cosine similarity analysis between baseline and anchor-prompt embeddings.
Covers Fig 3c and Fig S4 from the paper.

Paper Equation 13:  Cosine Similarity = (A · B) / (‖A‖ × ‖B‖)
where A and B are hidden-state embedding vectors for gene-gene interactions.

Paper results (Fig 3c):
  All classes stay > 0.98 across layers.
  Slight dip in deepest layers reflects task-relevant differentiation.

Paper results (Fig S4 — average across all gene-gene pairs):
  Early layers  (0-10)  : 0.925 - 0.935
  Middle layers (11-20) : ~0.955 peak
  Deep layers   (25-32) : decline 0.95 → 0.91
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Configs.config import RELATIONS
from Configs.paths_config import COSINE_SIM_CSV


def compute_classwise_cosine(
    emb_baseline: np.ndarray,
    emb_anchor:   np.ndarray,
    gt_labels:    pd.Series,
    output_csv:   str | None = None,
) -> pd.DataFrame:
    """
    Per-class and average cosine similarity between baseline and anchor
    hidden-state embeddings at every transformer layer.

    Parameters
    ----------
    emb_baseline : (num_layers, num_rows, hidden_dim) baseline embeddings
    emb_anchor   : (num_layers, num_rows, hidden_dim) anchor embeddings
    gt_labels    : Ground-truth relation labels (capitalised)
    output_csv   : Save path

    Returns
    -------
    DataFrame: Layer, Activation_sim, Inhibition_sim,
               Phosphorylation_sim, average_sim
    """
    num_layers = emb_baseline.shape[0]
    records: list[dict] = []

    for layer in range(num_layers):
        row: dict          = {"Layer": layer}
        layer_sims: list[float] = []

        for rel in RELATIONS:
            cap_rel = rel.capitalize()
            mask    = gt_labels == cap_rel
            if mask.sum() == 0:
                row[f"{rel}_sim"] = float("nan")
                continue
            b    = emb_baseline[layer][mask]
            a    = emb_anchor[layer][mask]
            sims = [float(cosine_similarity(b[i:i+1], a[i:i+1])[0, 0])
                    for i in range(len(b))]
            mean_sim          = float(np.mean(sims))
            row[f"{rel}_sim"] = mean_sim
            layer_sims.append(mean_sim)

        row["average_sim"] = float(np.mean(layer_sims)) if layer_sims else float("nan")
        records.append(row)

    sim_df    = pd.DataFrame(records)
    save_path = output_csv or str(COSINE_SIM_CSV)
    sim_df.to_csv(save_path, index=False)
    print(f"  ✔ Cosine similarity saved → {save_path}")
    return sim_df


def compute_sample_cosine_trajectory(
    emb_baseline: np.ndarray,
    emb_anchor:   np.ndarray,
    sample_index: int,
) -> np.ndarray:
    """
    Cosine similarity for one gene-pair sample across all layers.

    Returns
    -------
    1-D float32 array of shape (num_layers,)
    """
    num_layers = emb_baseline.shape[0]
    traj       = np.zeros(num_layers, dtype=np.float32)
    for layer in range(num_layers):
        b           = emb_baseline[layer][sample_index:sample_index + 1]
        a           = emb_anchor[layer][sample_index:sample_index + 1]
        traj[layer] = float(cosine_similarity(b, a)[0, 0])
    return traj

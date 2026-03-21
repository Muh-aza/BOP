"""
src/analysis/cosine_analysis.py
================================
Cosine similarity analysis between baseline and anchor-prompt embeddings
across all transformer layers of LLaMA-3.1-8B (33 layers total).

Cosine Similarity = (A · B) / (‖A‖ × ‖B‖)
where A and B are hidden-state embedding vectors for gene-gene interactions.

Functions
---------
compute_classwise_cosine        — per-class sim at every layer (main analysis)
compute_sample_cosine_trajectory — single gene-pair sim across all layers
compute_layerwise_cosine_matrix  — full N×N pairwise cosine matrix per layer
compute_intraclass_cosine        — avg pairwise sim within each class per layer
compute_interclass_cosine        — avg pairwise sim between classes per layer
summarise_cosine_by_split        — compare train/validation/test cosine profiles
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Configs.config import RELATIONS
from Configs.paths_config import COSINE_SIM_CSV


# ── 1. Per-class cosine similarity across all layers ──────────────────────────

def compute_classwise_cosine(
    emb_baseline: np.ndarray,
    emb_anchor:   np.ndarray,
    gt_labels:    pd.Series,
    output_csv:   str | None = None,
) -> pd.DataFrame:
    """
    Compute per-class and average cosine similarity between baseline and
    anchor hidden-state embeddings at every transformer layer (all 33).

    Parameters
    ----------
    emb_baseline : (num_layers, num_rows, hidden_dim)  baseline embeddings
    emb_anchor   : (num_layers, num_rows, hidden_dim)  anchor embeddings
    gt_labels    : Ground-truth relation labels (capitalised Series)
    output_csv   : Save path for results CSV

    Returns
    -------
    DataFrame with columns:
      Layer, layer_name,
      activation_sim, inhibition_sim, phosphorylation_sim,
      average_sim, num_samples
    """
    num_layers = emb_baseline.shape[0]
    layer_names = ["embedding"] + [f"block_{i:02d}" for i in range(1, num_layers)]
    records: list[dict] = []

    for layer in range(num_layers):
        row: dict = {
            "Layer":      layer,
            "layer_name": layer_names[layer],
        }
        layer_sims: list[float] = []

        for rel in RELATIONS:
            cap_rel = rel.capitalize()
            mask    = gt_labels == cap_rel
            n       = int(mask.sum())

            if n == 0:
                row[f"{rel}_sim"]     = float("nan")
                row[f"{rel}_n"]       = 0
                continue

            b = emb_baseline[layer][mask.values]
            a = emb_anchor[layer][mask.values]

            sims = [
                float(cosine_similarity(b[i:i+1], a[i:i+1])[0, 0])
                for i in range(n)
            ]
            mean_sim          = float(np.mean(sims))
            row[f"{rel}_sim"] = mean_sim
            row[f"{rel}_n"]   = n
            layer_sims.append(mean_sim)

        row["average_sim"]  = float(np.mean(layer_sims)) if layer_sims else float("nan")
        row["num_samples"]  = int(mask.shape[0]) if hasattr(mask, "shape") else len(gt_labels)
        records.append(row)

    sim_df    = pd.DataFrame(records)
    save_path = output_csv or str(COSINE_SIM_CSV)
    sim_df.to_csv(save_path, index=False)
    print(f"  ✔ Cosine similarity saved → {save_path}  ({num_layers} layers)")
    return sim_df


# ── 2. Per-sample cosine trajectory across all layers ─────────────────────────

def compute_sample_cosine_trajectory(
    emb_baseline: np.ndarray,
    emb_anchor:   np.ndarray,
    sample_index: int,
) -> np.ndarray:
    """
    Cosine similarity for one gene-pair sample across all 33 layers.

    Parameters
    ----------
    emb_baseline : (num_layers, num_rows, hidden_dim)
    emb_anchor   : (num_layers, num_rows, hidden_dim)
    sample_index : Row index of the gene-pair to trace

    Returns
    -------
    1-D float32 array of shape (num_layers,).
    Index 0 = embedding layer, 1-32 = transformer blocks.
    """
    num_layers = emb_baseline.shape[0]
    traj       = np.zeros(num_layers, dtype=np.float32)
    for layer in range(num_layers):
        b           = emb_baseline[layer][sample_index : sample_index + 1]
        a           = emb_anchor[layer][sample_index : sample_index + 1]
        traj[layer] = float(cosine_similarity(b, a)[0, 0])
    return traj


# ── 3. Full N×N pairwise cosine matrix at a single layer ──────────────────────

def compute_layerwise_cosine_matrix(
    embeddings: np.ndarray,
    layer:      int,
) -> np.ndarray:
    """
    Compute the full N×N pairwise cosine similarity matrix for all samples
    at a given layer.

    Parameters
    ----------
    embeddings : (num_layers, num_samples, hidden_dim)
    layer      : Which layer to compute the matrix for (0-32)

    Returns
    -------
    (num_samples, num_samples) float32 array.
    Entry [i, j] = cosine similarity between sample i and sample j.
    """
    layer_emb = embeddings[layer].astype(np.float32)
    return cosine_similarity(layer_emb).astype(np.float32)


# ── 4. Intra-class cosine similarity across all layers ────────────────────────

def compute_intraclass_cosine(
    embeddings: np.ndarray,
    gt_labels:  pd.Series,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Average pairwise cosine similarity within each relation class at every layer.
    High intra-class similarity → embeddings for the same relation are tightly
    clustered at that layer.

    Parameters
    ----------
    embeddings : (num_layers, num_samples, hidden_dim)
    gt_labels  : Ground-truth relation labels (capitalised)
    output_csv : Optional save path

    Returns
    -------
    DataFrame: Layer, layer_name, activation_intra, inhibition_intra,
               phosphorylation_intra, average_intra
    """
    num_layers  = embeddings.shape[0]
    layer_names = ["embedding"] + [f"block_{i:02d}" for i in range(1, num_layers)]
    records: list[dict] = []

    for layer in range(num_layers):
        row: dict = {"Layer": layer, "layer_name": layer_names[layer]}
        all_intra: list[float] = []

        for rel in RELATIONS:
            cap_rel = rel.capitalize()
            mask    = (gt_labels == cap_rel).values
            n       = int(mask.sum())

            if n < 2:
                row[f"{rel}_intra"] = float("nan")
                continue

            vecs = embeddings[layer][mask].astype(np.float32)
            sim_matrix = cosine_similarity(vecs)
            upper_tri  = sim_matrix[np.triu_indices(n, k=1)]
            mean_intra           = float(np.mean(upper_tri))
            row[f"{rel}_intra"]  = mean_intra
            all_intra.append(mean_intra)

        row["average_intra"] = float(np.mean(all_intra)) if all_intra else float("nan")
        records.append(row)

    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"  ✔ Intra-class cosine saved → {output_csv}")
    return df


# ── 5. Inter-class cosine similarity across all layers ────────────────────────

def compute_interclass_cosine(
    embeddings: np.ndarray,
    gt_labels:  pd.Series,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Average pairwise cosine similarity between different relation classes
    at every layer.
    Low inter-class similarity → the model separates classes well at that layer.

    Parameters
    ----------
    embeddings : (num_layers, num_samples, hidden_dim)
    gt_labels  : Ground-truth relation labels (capitalised)
    output_csv : Optional save path

    Returns
    -------
    DataFrame: Layer, layer_name,
               activation_vs_inhibition,
               activation_vs_phosphorylation,
               inhibition_vs_phosphorylation,
               average_inter
    """
    num_layers  = embeddings.shape[0]
    layer_names = ["embedding"] + [f"block_{i:02d}" for i in range(1, num_layers)]
    pairs       = [
        ("Activation",     "Inhibition"),
        ("Activation",     "Phosphorylation"),
        ("Inhibition",     "Phosphorylation"),
    ]
    pair_keys = [
        "activation_vs_inhibition",
        "activation_vs_phosphorylation",
        "inhibition_vs_phosphorylation",
    ]
    records: list[dict] = []

    for layer in range(num_layers):
        row: dict = {"Layer": layer, "layer_name": layer_names[layer]}
        all_inter: list[float] = []

        for (rel_a, rel_b), key in zip(pairs, pair_keys):
            mask_a = (gt_labels == rel_a).values
            mask_b = (gt_labels == rel_b).values

            if mask_a.sum() == 0 or mask_b.sum() == 0:
                row[key] = float("nan")
                continue

            vecs_a     = embeddings[layer][mask_a].astype(np.float32)
            vecs_b     = embeddings[layer][mask_b].astype(np.float32)
            sim_matrix = cosine_similarity(vecs_a, vecs_b)
            mean_inter = float(np.mean(sim_matrix))
            row[key]   = mean_inter
            all_inter.append(mean_inter)

        row["average_inter"] = float(np.mean(all_inter)) if all_inter else float("nan")
        records.append(row)

    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"  ✔ Inter-class cosine saved → {output_csv}")
    return df


# ── 6. Compare cosine profiles across train/validation/test splits ─────────────

def summarise_cosine_by_split(
    split_results: dict[str, pd.DataFrame],
    output_csv:    str | None = None,
) -> pd.DataFrame:
    """
    Combine per-split cosine DataFrames into a single comparison table.
    Useful for checking whether embedding geometry is consistent across splits.

    Parameters
    ----------
    split_results : {'train': df, 'validation': df, 'test': df}
                    Each df is the output of compute_classwise_cosine()
    output_csv    : Optional save path

    Returns
    -------
    DataFrame with columns:
      Layer, layer_name,
      train_avg_sim, validation_avg_sim, test_avg_sim,
      train_val_delta, train_test_delta
    """
    records: list[dict] = []

    splits    = list(split_results.keys())
    first_df  = next(iter(split_results.values()))
    num_layers = len(first_df)

    for i in range(num_layers):
        row: dict = {
            "Layer":      int(first_df.iloc[i]["Layer"]),
            "layer_name": str(first_df.iloc[i]["layer_name"]),
        }
        avgs: dict[str, float] = {}
        for split, df in split_results.items():
            avg = float(df.iloc[i].get("average_sim", float("nan")))
            row[f"{split}_avg_sim"] = avg
            avgs[split] = avg

        if "train" in avgs and "validation" in avgs:
            row["train_val_delta"] = round(avgs["train"] - avgs["validation"], 6)
        if "train" in avgs and "test" in avgs:
            row["train_test_delta"] = round(avgs["train"] - avgs["test"], 6)

        records.append(row)

    summary_df = pd.DataFrame(records)
    if output_csv:
        summary_df.to_csv(output_csv, index=False)
        print(f"  ✔ Split cosine summary saved → {output_csv}")
    return summary_df

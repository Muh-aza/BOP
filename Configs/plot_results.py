"""
src/analysis/plot_results.py
=============================
All result plots NOT covered by umap_analysis.py:
  - Convergence curves: train vs val F1 across iterations  (Fig 2a / 2c / S1)
  - Per-class cosine similarity across layers              (Fig 3c)
  - Average cosine similarity across all gene-gene pairs   (Fig S4)
  - Per-sample cosine trajectory with gene labels          (Extra)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Configs.config import COLORS, MARKERS, PLOT_DPI, RELATIONS
from Configs.paths_config import CONVERGENCE_PNG, COSINE_CLASS_PNG, COSINE_AVG_PNG, COSINE_TRAJ_PNG, TRAJECTORY_XLS
from src.analysis.cosine_analysis import compute_sample_cosine_trajectory


# ── Convergence curve (Fig 2a / 2c) ───────────────────────────────────────────
def plot_convergence(
    study_df:   pd.DataFrame,
    model_name: str,
    output_png: str | None = None,
) -> None:
    """
    Train vs validation macro-F1 across optimisation iterations.

    Parameters
    ----------
    study_df   : DataFrame with columns trial_number, train_macro_f1, val_macro_f1
    model_name : Used in the plot title
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(study_df["trial_number"], study_df["train_macro_f1"],
            label="Training F₁",   color="steelblue",  linewidth=2)
    ax.plot(study_df["trial_number"], study_df["val_macro_f1"],
            label="Validation F₁", color="darkorange", linewidth=2, linestyle="--")
    ax.set_xlabel("Optimisation Iteration", fontsize=12)
    ax.set_ylabel("Macro-averaged F₁",      fontsize=12)
    ax.set_title(f"BOP Convergence — {model_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = output_png or str(CONVERGENCE_PNG)
    plt.savefig(path, dpi=PLOT_DPI)
    plt.show()
    print(f"  ✔ Convergence curve saved → {path}")


# ── Per-class cosine similarity (Fig 3c) ──────────────────────────────────────
def plot_classwise_cosine(
    sim_df:     pd.DataFrame,
    output_png: str | None = None,
) -> None:
    """Line plot of per-class cosine similarity across transformer layers."""
    rel_colors = {
        "activation": "steelblue",
        "inhibition": "firebrick",
        "phosphorylation": "forestgreen",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for rel in RELATIONS:
        col = f"{rel}_sim"
        if col in sim_df.columns:
            ax.plot(sim_df["Layer"], sim_df[col],
                    label=rel.capitalize(), color=rel_colors[rel], linewidth=2)
    ax.axhline(0.98, color="gray", linestyle="--", linewidth=1, label=">0.98 threshold")
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity",  fontsize=12)
    ax.set_title("Class-wise Cosine Similarity: Baseline vs Structural Anchor", fontsize=13)
    ax.legend(fontsize=11); ax.set_ylim(0.85, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = output_png or str(COSINE_CLASS_PNG)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Class-wise cosine saved → {path}")


# ── Average cosine similarity (Fig S4) ────────────────────────────────────────
def plot_average_cosine(
    sim_df:     pd.DataFrame,
    output_png: str | None = None,
) -> None:
    """
    Average cosine similarity across all gene-gene pairs per layer.
    Paper: Early=0.925-0.935, Middle=~0.955, Deep=0.91-0.95.
    """
    n = len(sim_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim_df["Layer"], sim_df["average_sim"],
            color="darkorchid", linewidth=2.5, marker="o", markersize=4)
    ax.axvspan(0,  10, alpha=0.07, color="blue",  label="Early  (0-10):  0.925-0.935")
    ax.axvspan(11, 20, alpha=0.07, color="green", label="Middle (11-20): ~0.955")
    ax.axvspan(25, n,  alpha=0.07, color="red",   label="Deep   (25-32): 0.91-0.95")
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Average Cosine Similarity", fontsize=12)
    ax.set_title("Average Cosine Similarity — All Gene–Gene Pairs\n"
                 "(Baseline vs Structural Anchor)", fontsize=13)
    ax.legend(fontsize=10); ax.set_ylim(0.85, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = output_png or str(COSINE_AVG_PNG)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Average cosine saved → {path}")


# ── Per-sample cosine trajectory (gene-pair, layer by layer) ──────────────────
def plot_cosine_trajectories(
    emb_baseline:   np.ndarray,
    emb_anchor:     np.ndarray,
    df:             pd.DataFrame,
    sample_indices: dict[str, int],
    output_xlsx:    str | None = None,
    output_png:     str | None = None,
) -> pd.DataFrame:
    """
    Show how cosine similarity between baseline and anchor evolves
    layer-by-layer for one representative sample per relation class.

    Parameters
    ----------
    sample_indices : {relation_label: row_index}
                     e.g. {"Activation": 5, "Inhibition": 37, "Phosphorylation": 65}
    """
    relations  = list(sample_indices.keys())
    layers_arr = np.arange(emb_baseline.shape[0])
    fig, axes  = plt.subplots(1, len(relations), figsize=(7 * len(relations), 6))
    if len(relations) == 1:
        axes = [axes]

    all_trajs: dict[str, np.ndarray] = {}

    for ax, rel in zip(axes, relations):
        idx  = sample_indices[rel]
        traj = compute_sample_cosine_trajectory(emb_baseline, emb_anchor, idx)
        all_trajs[rel] = traj

        color = COLORS.get(rel, "gray")
        ax.plot(layers_arr, traj, color=color, linewidth=2.5, marker="o", markersize=4)
        ax.axhline(0.98, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_between(layers_arr, traj, 0.98,
                        where=(traj < 0.98), alpha=0.15, color=color)

        # Annotate min / max
        min_l = int(np.argmin(traj)); max_l = int(np.argmax(traj))
        ax.annotate(f"min={traj[min_l]:.3f}", xy=(min_l, traj[min_l]),
                    xytext=(min_l + 1, traj[min_l] - 0.01), fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        ax.annotate(f"max={traj[max_l]:.3f}", xy=(max_l, traj[max_l]),
                    xytext=(max_l + 1, traj[max_l] + 0.005), fontsize=8, color="green",
                    arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

        gene1 = str(df.iloc[idx].get("Gene-A", idx))
        gene2 = str(df.iloc[idx].get("Gene-B", ""))
        ax.set_title(f"{rel}\n{gene1} → {gene2}", fontsize=13)
        ax.set_xlabel("Transformer Layer", fontsize=11)
        ax.set_ylabel("Cosine Similarity",  fontsize=11)
        ax.set_ylim(0.85, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Gene-Pair Cosine Trajectory — Baseline vs Structural Anchor\n"
                 "(LLaMA-3.1-8B)", fontsize=14)
    plt.tight_layout()
    path = output_png or str(COSINE_TRAJ_PNG)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Trajectory plot saved → {path}")

    # Save to Excel
    traj_df = pd.DataFrame({"Layer": layers_arr})
    for rel, traj in all_trajs.items():
        traj_df[f"{rel}_cosine"] = traj

    xlsx_path = output_xlsx or str(TRAJECTORY_XLS)
    with pd.ExcelWriter(xlsx_path) as writer:
        traj_df.to_excel(writer, sheet_name="All_Relations", index=False)
        for rel, traj in all_trajs.items():
            pd.DataFrame({"Layer": layers_arr, "Cosine": traj}
                         ).to_excel(writer, sheet_name=rel, index=False)
    print(f"  ✔ Trajectory data saved → {xlsx_path}")
    return traj_df

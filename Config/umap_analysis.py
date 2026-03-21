"""
src/analysis/umap_analysis.py
==============================
UMAP-based visualisation of hidden-state embeddings across transformer layers.

UMAP settings:
  n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from Configs.config import (
    MARKERS, COLORS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    UMAP_METRIC, UMAP_RANDOM_STATE, UMAP_GRID_COLS, PLOT_DPI,
)

RELATIONS = ["Activation", "Inhibition", "Phosphorylation"]


def _fit_umap(x_scaled: np.ndarray) -> np.ndarray:
    return UMAP(
        n_components = 2,
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = UMAP_RANDOM_STATE,
    ).fit_transform(x_scaled)


def plot_umap_grid(
    emb:        np.ndarray,
    df:         pd.DataFrame,
    ch_vals:    list[float],
    title:      str = "",
    output_png: str | None = None,
) -> None:
    """
    5-column UMAP grid — one panel per transformer layer.

    Parameters
    ----------
    emb        : (num_layers, num_rows, hidden_dim)
    df         : DataFrame with 'Ground truth' and 'Label' columns
    ch_vals    : CHI per layer from chi_analysis.compute_chi_per_layer()
    title      : Suptitle for the grid
    output_png : Save path at 600 dpi
    """
    num_layers = emb.shape[0]
    grid_rows  = math.ceil((num_layers - 1) / UMAP_GRID_COLS)
    fig        = plt.figure(figsize=(UMAP_GRID_COLS * 8, grid_rows * 7))
    if title:
        fig.suptitle(title, fontsize=14, y=1.005)

    for idx, layer in enumerate(range(1, num_layers)):
        pts = _fit_umap(StandardScaler().fit_transform(emb[layer]))
        ax  = fig.add_subplot(grid_rows, UMAP_GRID_COLS, idx + 1)

        for rel in RELATIONS:
            mask = (df["Ground truth"] == rel) & (df["Label"] == "Correct")
            ax.scatter(pts[mask, 0], pts[mask, 1],
                       marker=MARKERS[rel], c=COLORS[rel],
                       alpha=0.8, s=20, label=rel if idx == 0 else None)

        wrong = df["Label"] == "Incorrect"
        ax.scatter(pts[wrong, 0], pts[wrong, 1],
                   marker=MARKERS["Incorrect"], c=COLORS["Incorrect"],
                   alpha=0.6, s=20, label="Incorrect" if idx == 0 else None)

        ax.set_title(f"Layer {layer} | CH={ch_vals[layer - 1]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11)
    plt.subplots_adjust(bottom=0.05)
    plt.tight_layout()
    if output_png:
        plt.savefig(output_png, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  ✔ UMAP grid saved → {output_png}")
    plt.show()


def plot_umap_comparison(
    emb_baseline: np.ndarray,
    emb_anchor:   np.ndarray,
    df:           pd.DataFrame,
    layer:        int = 32,
    output_png:   str | None = None,
) -> None:
    """
    Side-by-side UMAP: baseline vs structural anchor at a chosen layer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, emb, ttl in zip(
        axes,
        [emb_baseline, emb_anchor],
        ["Baseline Prompt (P)", "Structural Anchor Prompt (P+1)"],
    ):
        pts = _fit_umap(StandardScaler().fit_transform(emb[layer]))
        for rel in RELATIONS:
            mask = df["Ground truth"] == rel
            ax.scatter(pts[mask, 0], pts[mask, 1],
                       marker=MARKERS[rel], c=COLORS[rel],
                       alpha=0.85, s=40, label=rel)
        ax.set_title(f"{ttl} — Layer {layer}", fontsize=13)
        ax.set_xlabel("UMAP-1", fontsize=11); ax.set_ylabel("UMAP-2", fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("UMAP Comparison — Baseline vs Structural Anchor (LLaMA-3.1-8B)",
                 fontsize=14)
    plt.tight_layout()
    if output_png:
        plt.savefig(output_png, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  ✔ UMAP comparison saved → {output_png}")
    plt.show()

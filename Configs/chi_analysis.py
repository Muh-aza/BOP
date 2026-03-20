"""
src/analysis/chi_analysis.py
=============================
Calinski-Harabász Index (CHI) per transformer layer — paper Fig 3b / Eq. 12.

CHI = [Tr(Bk) / Tr(Wk)] × [(n-k) / (k-1)]

where Bk = between-class scatter, Wk = within-class scatter.
Higher CHI → better-separated, more compact clusters.

Paper results:
  Baseline (layer 33) : CHI ≈ 66.9
  Anchor   (layer 33) : CHI ≈ 86.9  (2.28× improvement)
  Linear trendline slopes: anchor=1.12 vs baseline=0.48
"""

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from Configs.paths_config import CH_VALUES_CSV


def compute_chi_per_layer(
    emb:        np.ndarray,
    gt_labels:  pd.Series,
    output_csv: str | None = None,
) -> tuple[list[float], pd.DataFrame]:
    """
    Compute CHI for layers 1 … N (layer 0 is the embedding layer — skipped).

    Parameters
    ----------
    emb        : (num_layers, num_rows, hidden_dim) ndarray
    gt_labels  : Ground-truth relation labels (Series, capitalised values)
    output_csv : Save path for CHI table CSV

    Returns
    -------
    (ch_vals, ch_df) — ch_vals is a list, ch_df has columns Layer and CH
    """
    gt_codes   = gt_labels.astype("category").cat.codes
    num_layers = emb.shape[0]
    ch_vals: list[float] = []

    for layer in range(1, num_layers):
        x_scaled = StandardScaler().fit_transform(emb[layer])
        ch_vals.append(calinski_harabasz_score(x_scaled, gt_codes))

    ch_df     = pd.DataFrame({"Layer": range(1, num_layers), "CH": ch_vals})
    save_path = output_csv or str(CH_VALUES_CSV)
    ch_df.to_csv(save_path, index=False)
    print(f"  ✔ CHI saved → {save_path}  (terminal={ch_vals[-1]:.1f})")
    return ch_vals, ch_df

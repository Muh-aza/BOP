"""
src/scripts/run_embedding_analysis.py
======================================
Full embedding analysis pipeline using LLaMA-3.1-8B-Instruct.
Runs separately on training, validation, and test splits so that
embedding geometry and cosine similarity results are kept independent
per split rather than merged into a single output.

Outputs produced per split (Result/train/, Result/validation/, Result/test/)
-----------------------------------------------------------------------------
  - CHI per layer: baseline vs anchor
  - Per-class cosine similarity across layers
  - Average cosine similarity across all gene-gene pairs
  - Per-sample cosine trajectory
  - UMAP grid — anchor prompt
  - UMAP grid — baseline prompt
  - Side-by-side UMAP comparison at chosen layer

Usage
-----
# All three splits (default):
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U"

# Single split only:
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --splits test
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --splits train validation

# Skip baseline extraction (faster):
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --skip_baseline

# Change UMAP comparison layer:
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --layer 32
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from Configs.paths_config import (
    ensure_dirs, split_paths,
    get_training_csv, get_validation_csv, get_testing_xls,
    RESULTS_DIR,
)
from Configs.prompt_config import assemble_system_prompt, BASELINE_SAP
from src.models.llama_backend import LlamaBackend
from src.analysis.chi_analysis import compute_chi_per_layer
from src.analysis.cosine_analysis import (
    compute_classwise_cosine,
    compute_intraclass_cosine,
    compute_interclass_cosine,
    summarise_cosine_by_split,
)
from src.analysis.umap_analysis import plot_umap_grid, plot_umap_comparison
from src.analysis.plot_results import (
    plot_classwise_cosine, plot_average_cosine, plot_cosine_trajectories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BOP embedding analysis — train / validation / test splits."
    )
    p.add_argument(
        "--anchor_key", type=str, default="F3eI?%qt,NbnG8U",
        help="Best 15-char ASCII structural anchor from optimisation.",
    )
    p.add_argument(
        "--splits", nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Which splits to run (default: all three).",
    )
    p.add_argument(
        "--layer", type=int, default=32,
        help="Transformer layer for UMAP side-by-side comparison (default: 32).",
    )
    p.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip baseline embedding extraction to save time.",
    )
    return p.parse_args()


def _load_split(split: str) -> pd.DataFrame:
    """Load the correct file for a given split name."""
    if split == "train":
        path = str(get_training_csv())
        return pd.read_csv(path)
    elif split == "validation":
        path = str(get_validation_csv())
        return pd.read_csv(path)
    else:
        path = str(get_testing_xls())
        return pd.read_excel(path)


def _run_predictions(
    backend: LlamaBackend, df: pd.DataFrame, system: str, split: str,
) -> pd.DataFrame:
    """Add Prediction and Label columns using the given system prompt."""
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting [{split}]"):
        gene1 = str(row.get("Gene-A", row.get("starter", ""))).upper()
        gene2 = str(row.get("Gene-B", row.get("receiver", ""))).upper()
        user_q = f"What relationship exists between gene {gene1} and gene {gene2}?"
        _, pred = backend.query(system, user_q)
        preds.append(
            pred.capitalize() if pred and pred != "no information"
            else "No information"
        )
    df = df.copy()
    df["Prediction"] = preds
    gt_col = "Ground truth" if "Ground truth" in df.columns else "relation_name"
    df["Label"] = np.where(df["Prediction"] == df[gt_col], "Correct", "Incorrect")
    df["_gt_col"] = gt_col
    return df


def _make_texts(df: pd.DataFrame, mode: str = "anchor") -> list[str]:
    """Build embedding input strings from gene-pair records."""
    gt_col = df["_gt_col"].iloc[0] if "_gt_col" in df.columns else "Ground truth"
    geneA  = "Gene-A" if "Gene-A" in df.columns else "starter"
    geneB  = "Gene-B" if "Gene-B" in df.columns else "receiver"
    return [
        f"{r[geneA]} interacts with {r[geneB]}, "
        f"ground truth is {r[gt_col]}, "
        f"prediction is {r.get('Prediction', mode)}"
        for _, r in df.iterrows()
    ]


def _gt_series(df: pd.DataFrame) -> pd.Series:
    """Return the ground-truth label column regardless of column name."""
    col = df["_gt_col"].iloc[0] if "_gt_col" in df.columns else "Ground truth"
    return df[col]


def _pick_samples(df: pd.DataFrame) -> dict[str, int]:
    """Pick one representative correct sample per relation class."""
    gt = _gt_series(df)
    indices = {}
    for rel in ["Activation", "Inhibition", "Phosphorylation"]:
        correct = df[(gt == rel) & (df.get("Label", pd.Series()) == "Correct")]
        indices[rel] = int(
            correct.index[0] if len(correct) > 0
            else df[gt == rel].index[0]
        )
    return indices


def _run_split(
    split:         str,
    df:            pd.DataFrame,
    backend:       LlamaBackend,
    anchor_sys:    str,
    baseline_sys:  str,
    layer:         int,
    skip_baseline: bool,
) -> pd.DataFrame:
    """Run the full embedding analysis pipeline for one split. Returns sim_df."""
    paths = split_paths(split)
    os.makedirs(paths["dir"], exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Split: {split.upper()}  ({len(df)} samples)")
    print(f"{'='*55}")

    # ── Predictions ────────────────────────────────────────────────────────────
    df = _run_predictions(backend, df, anchor_sys, split)
    acc = (df["Label"] == "Correct").mean()
    print(f"  Accuracy: {acc:.3f}")

    # ── Extract anchor embeddings ──────────────────────────────────────────────
    print(f"\n  Extracting anchor-prompt embeddings [{split}] ...")
    emb_anchor = backend.build_layer_embeddings(_make_texts(df, "anchor"))
    print(f"  Shape: {emb_anchor.shape}  [layers × samples × dim]")

    if not skip_baseline:
        print(f"  Extracting baseline-prompt embeddings [{split}] ...")
        emb_baseline = backend.build_layer_embeddings(_make_texts(df, "baseline"))
    else:
        print(f"  Using anchor as baseline proxy (--skip_baseline).")
        emb_baseline = emb_anchor

    gt = _gt_series(df)

    # ── CHI per layer ──────────────────────────────────────────────────────────
    print(f"\n  Computing CHI — anchor [{split}] ...")
    ch_vals_anchor, _ = compute_chi_per_layer(
        emb_anchor, gt,
        output_csv=str(paths["ch_anchor_csv"]),
    )
    print(f"  Terminal CHI (anchor): {ch_vals_anchor[-1]:.1f}")

    if not skip_baseline:
        print(f"  Computing CHI — baseline [{split}] ...")
        ch_vals_base, _ = compute_chi_per_layer(
            emb_baseline, gt,
            output_csv=str(paths["ch_baseline_csv"]),
        )
        print(f"  Terminal CHI (baseline): {ch_vals_base[-1]:.1f}")
    else:
        ch_vals_base = ch_vals_anchor

    # ── Cosine similarity ──────────────────────────────────────────────────────
    print(f"\n  Computing cosine similarity [{split}] ...")
    sim_df = compute_classwise_cosine(
        emb_baseline, emb_anchor, gt,
        output_csv=str(paths["cosine_csv"]),
    )
    print(f"  Middle-layer avg cosine: {sim_df['average_sim'].iloc[11:21].mean():.4f}")

    plot_classwise_cosine(sim_df, output_png=str(paths["cosine_class_png"]))
    plot_average_cosine(  sim_df, output_png=str(paths["cosine_avg_png"]))

    # ── Intra-class cosine similarity (all 33 layers) ──────────────────────────
    print(f"\n  Computing intra-class cosine [{split}] ...")
    compute_intraclass_cosine(
        emb_anchor, gt,
        output_csv=str(paths["dir"] / "cosine_intraclass.csv"),
    )

    # ── Inter-class cosine similarity (all 33 layers) ──────────────────────────
    print(f"  Computing inter-class cosine [{split}] ...")
    compute_interclass_cosine(
        emb_anchor, gt,
        output_csv=str(paths["dir"] / "cosine_interclass.csv"),
    )

    # ── Cosine trajectory ──────────────────────────────────────────────────────
    print(f"\n  Computing cosine trajectory [{split}] ...")
    sample_indices = _pick_samples(df)
    plot_cosine_trajectories(
        emb_baseline, emb_anchor, df, sample_indices,
        output_xlsx = str(paths["trajectory_xlsx"]),
        output_png  = str(paths["cosine_traj_png"]),
    )

    # ── UMAP grid — anchor ─────────────────────────────────────────────────────
    print(f"\n  Generating UMAP grid — anchor [{split}] ...")
    plot_umap_grid(
        emb_anchor, df, ch_vals_anchor,
        title      = f"UMAP — Structural Anchor ({split.capitalize()})",
        output_png = str(paths["umap_anchor_png"]),
    )

    if not skip_baseline:
        print(f"  Generating UMAP grid — baseline [{split}] ...")
        plot_umap_grid(
            emb_baseline, df, ch_vals_base,
            title      = f"UMAP — Baseline Prompt ({split.capitalize()})",
            output_png = str(paths["umap_base_png"]),
        )

        print(f"  Generating UMAP comparison at layer {layer} [{split}] ...")
        plot_umap_comparison(
            emb_baseline, emb_anchor, df,
            layer      = layer,
            output_png = str(paths["umap_compare_png"]),
        )

    print(f"\n  ✅  {split.upper()} outputs saved → {paths['dir']}/")
    return sim_df


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print(f"\n{'='*55}")
    print(f"  BOP Embedding Analysis  (LLaMA-3.1-8B)")
    print(f"  Anchor key   : {args.anchor_key}")
    print(f"  Splits       : {', '.join(args.splits)}")
    print(f"  Layer        : {args.layer}")
    print(f"  Skip baseline: {args.skip_baseline}")
    print(f"{'='*55}")

    # ── Load LLaMA once — reuse across all splits ──────────────────────────────
    backend = LlamaBackend()

    anchor_sys = assemble_system_prompt(
        BASELINE_SAP.role, BASELINE_SAP.aims,
        BASELINE_SAP.description, args.anchor_key,
    )
    baseline_sys = assemble_system_prompt(
        BASELINE_SAP.role, BASELINE_SAP.aims,
        BASELINE_SAP.description, "",
    )

    # ── Run each requested split independently ─────────────────────────────────
    split_cosine_dfs: dict[str, pd.DataFrame] = {}

    for split in args.splits:
        print(f"\nLoading {split} data ...")
        df = _load_split(split)
        print(f"  {len(df)} samples loaded.")

        sim_df = _run_split(
            split         = split,
            df            = df,
            backend       = backend,
            anchor_sys    = anchor_sys,
            baseline_sys  = baseline_sys,
            layer         = args.layer,
            skip_baseline = args.skip_baseline,
        )
        split_cosine_dfs[split] = sim_df

    # ── Cross-split cosine summary (if more than one split was run) ────────────
    if len(split_cosine_dfs) > 1:
        print("\nGenerating cross-split cosine summary ...")
        summarise_cosine_by_split(
            split_cosine_dfs,
            output_csv=str(RESULTS_DIR / "cosine_split_summary.csv"),
        )

    print(f"\n{'='*55}")
    print(f"  All splits complete.")
    print(f"  Results saved under: {RESULTS_DIR}/")
    print(f"    train/      — training split outputs")
    print(f"    validation/ — validation split outputs")
    print(f"    test/       — test split outputs")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

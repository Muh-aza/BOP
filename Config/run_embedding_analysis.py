"""
src/scripts/run_embedding_analysis.py
======================================
Full embedding analysis pipeline using LLaMA-3.1-8B-Instruct.
Produces UMAP visualisations, CHI, and cosine similarity analyses.

Outputs produced
----------------
  - CHI per layer: baseline vs anchor
  - Per-class cosine similarity across layers
  - Side-by-side UMAP comparison at best layer
  - UMAP grid (baseline prompt)
  - UMAP grid (structural anchor prompt)
  - Average cosine similarity across all gene-gene pairs
  - Per-sample cosine trajectory (Activation/Inhibition/Phosphorylation)

Usage
-----
# Full pipeline (baseline + anchor):
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U"

# Anchor-only (faster, skips baseline extraction):
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --skip_baseline

# Change comparison layer:
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --layer 32
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from Configs.config import RELATIONS
from Configs.paths_config import (
    ensure_dirs, get_testing_xls,
    CH_VALUES_CSV, COSINE_SIM_CSV, TRAJECTORY_XLS,
    RESULTS_DIR, UMAP_ANCHOR_PNG, UMAP_BASELINE_PNG,
    UMAP_COMPARE_PNG, COSINE_CLASS_PNG, COSINE_AVG_PNG, COSINE_TRAJ_PNG,
)
from Configs.prompt_config import assemble_system_prompt, BASELINE_SAP
from src.models.llama_backend import LlamaBackend
from src.analysis.chi_analysis import compute_chi_per_layer
from src.analysis.cosine_analysis import compute_classwise_cosine
from src.analysis.umap_analysis import plot_umap_grid, plot_umap_comparison
from src.analysis.plot_results import (
    plot_classwise_cosine, plot_average_cosine, plot_cosine_trajectories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BOP embedding analysis — UMAP + cosine similarity."
    )
    p.add_argument(
        "--anchor_key", type=str, default="F3eI?%qt,NbnG8U",
        help="Best 15-char ASCII structural anchor from optimisation.",
    )
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument(
        "--layer", type=int, default=32,
        help="Transformer layer for UMAP side-by-side comparison (default: 32).",
    )
    p.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip baseline embedding extraction to save time.",
    )
    return p.parse_args()


def _run_predictions(
    backend: LlamaBackend, df: pd.DataFrame, system: str,
) -> pd.DataFrame:
    """Add Prediction and Label columns using the given system prompt."""
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        user_q = (f"What relationship exists between gene "
                  f"{row['Gene-A']} and gene {row['Gene-B']}?")
        _, pred = backend.query(system, user_q)
        preds.append(pred.capitalize() if pred and pred != "no information"
                     else "No information")
    df = df.copy()
    df["Prediction"] = preds
    df["Label"]      = np.where(
        df["Prediction"] == df["Ground truth"], "Correct", "Incorrect"
    )
    return df


def _make_texts(df: pd.DataFrame, mode: str = "anchor") -> list[str]:
    """Build embedding input strings from gene-pair records."""
    return [
        f"{r['Gene-A']} interacts with {r['Gene-B']}, "
        f"ground truth is {r['Ground truth']}, "
        f"prediction is {r.get('Prediction', mode)}"
        for _, r in df.iterrows()
    ]


def _pick_samples(df: pd.DataFrame) -> dict[str, int]:
    """Pick one representative correct sample per relation class."""
    indices = {}
    for rel in ["Activation", "Inhibition", "Phosphorylation"]:
        correct = df[(df["Ground truth"] == rel) & (df["Label"] == "Correct")]
        indices[rel] = int(
            correct.index[0] if len(correct) > 0
            else df[df["Ground truth"] == rel].index[0]
        )
    return indices


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print(f"\n{'='*55}")
    print(f"  BOP Embedding Analysis  (LLaMA-3.1-8B)")
    print(f"  Anchor key   : {args.anchor_key}")
    print(f"  Layer        : {args.layer}")
    print(f"  Skip baseline: {args.skip_baseline}")
    print(f"{'='*55}\n")

    # ── Load test data ─────────────────────────────────────────────────────────
    test_path = args.test_file or str(get_testing_xls())
    df        = pd.read_excel(test_path)
    print(f"Loaded {len(df)} samples from {test_path}")

    # ── Load LLaMA-3.1 ────────────────────────────────────────────────────────
    backend = LlamaBackend()

    # ── Build system prompts ───────────────────────────────────────────────────
    anchor_sys   = assemble_system_prompt(
        BASELINE_SAP.role, BASELINE_SAP.aims,
        BASELINE_SAP.description, args.anchor_key,
    )
    baseline_sys = assemble_system_prompt(
        BASELINE_SAP.role, BASELINE_SAP.aims,
        BASELINE_SAP.description, "",
    )

    # ── Run predictions (anchor prompt) ───────────────────────────────────────
    print("\nRunning predictions (anchor prompt) ...")
    df  = _run_predictions(backend, df, anchor_sys)
    acc = (df["Label"] == "Correct").mean()
    print(f"  Accuracy: {acc:.3f}")

    # ── Extract embeddings ─────────────────────────────────────────────────────
    print("\nExtracting anchor-prompt hidden states ...")
    emb_anchor = backend.build_layer_embeddings(_make_texts(df, "anchor"))
    print(f"  Shape: {emb_anchor.shape}  [layers × samples × dim]")

    if not args.skip_baseline:
        print("\nExtracting baseline-prompt hidden states ...")
        emb_baseline = backend.build_layer_embeddings(_make_texts(df, "baseline"))
    else:
        print("\nUsing anchor embeddings as baseline proxy (--skip_baseline).")
        emb_baseline = emb_anchor

    # ── CHI per layer ──────────────────────────────────────────────────────────
    print("\nComputing CHI — anchor prompt ...")
    ch_vals_anchor, _ = compute_chi_per_layer(
        emb_anchor, df["Ground truth"],
        output_csv=str(CH_VALUES_CSV),
    )
    print(f"  Terminal CHI (anchor): {ch_vals_anchor[-1]:.1f}")

    if not args.skip_baseline:
        print("Computing CHI — baseline prompt ...")
        ch_vals_base, _ = compute_chi_per_layer(
            emb_baseline, df["Ground truth"],
            output_csv=str(RESULTS_DIR / "CH_values_baseline.csv"),
        )
        print(f"  Terminal CHI (baseline): {ch_vals_base[-1]:.1f}")

    # ── Per-class cosine similarity ────────────────────────────────────────────
    print("\nComputing class-wise cosine similarity ...")
    sim_df = compute_classwise_cosine(
        emb_baseline, emb_anchor, df["Ground truth"],
        output_csv=str(COSINE_SIM_CSV),
    )
    mid_sim = sim_df["average_sim"].iloc[11:21].mean()
    print(f"  Middle-layer avg cosine: {mid_sim:.4f}")

    plot_classwise_cosine(sim_df, output_png=str(COSINE_CLASS_PNG))
    plot_average_cosine(  sim_df, output_png=str(COSINE_AVG_PNG))

    # ── Cosine trajectory per gene-pair ───────────────────────────────────────
    print("\nComputing cosine trajectory per gene-pair ...")
    sample_indices = _pick_samples(df)
    print(f"  Samples: {sample_indices}")
    plot_cosine_trajectories(
        emb_baseline, emb_anchor, df, sample_indices,
        output_xlsx = str(TRAJECTORY_XLS),
        output_png  = str(COSINE_TRAJ_PNG),
    )

    # ── UMAP grid — anchor prompt ──────────────────────────────────────────────
    print("\nGenerating UMAP grid — Structural Anchor ...")
    plot_umap_grid(
        emb_anchor, df, ch_vals_anchor,
        title      = "UMAP Grid — Structural Anchor Prompt (LLaMA-3.1-8B)",
        output_png = str(UMAP_ANCHOR_PNG),
    )

    if not args.skip_baseline:
        # ── UMAP grid — baseline prompt ────────────────────────────────────────
        print("\nGenerating UMAP grid — Baseline Prompt ...")
        plot_umap_grid(
            emb_baseline, df, ch_vals_base,
            title      = "UMAP Grid — Baseline Prompt (LLaMA-3.1-8B)",
            output_png = str(UMAP_BASELINE_PNG),
        )

        # ── UMAP side-by-side comparison ───────────────────────────────────────
        print(f"\nGenerating UMAP comparison at layer {args.layer} ...")
        plot_umap_comparison(
            emb_baseline, emb_anchor, df,
            layer      = args.layer,
            output_png = str(UMAP_COMPARE_PNG),
        )

    print(f"\n✅  All outputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

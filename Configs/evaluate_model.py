"""
src/scripts/evaluate_model.py
==============================
Evaluate the best BOP prompt on the held-out test set.
Reads the Best Trial sheet from optimization_results.xlsx,
runs inference, and reports macro-F1, per-class F1, MCC,
precision and recall (paper Table / Fig 2 / Fig 4).

Usage
-----
python src/scripts/evaluate_model.py --model gpt-4o
python src/scripts/evaluate_model.py --model llama3
python src/scripts/evaluate_model.py --model cohere
"""

import argparse
import pandas as pd

from Configs.config import NO_INFO
from Configs.paths_config import (
    ensure_dirs, get_testing_xls,
    OPT_RESULTS_XLS, EVAL_RESULTS_XLS,
)
from Configs.prompt_config import assemble_system_prompt
from src.models.model_factory import build_backend
from src.utils.metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BOP — evaluate best prompt on held-out test set."
    )
    p.add_argument(
        "--model", type=str, default="gpt-4o",
        choices=["gpt-3.5", "gpt-4", "gpt-4o", "cohere", "llama3"],
    )
    p.add_argument(
        "--opt_results", type=str, default=None,
        help="Path to optimization_results.xlsx (reads Best Trial sheet).",
    )
    p.add_argument("--test_file",   type=str, default=None)
    p.add_argument("--output_path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    # ── Load best prompt from optimisation results ─────────────────────────────
    opt_path = args.opt_results or str(OPT_RESULTS_XLS)
    best     = pd.read_excel(opt_path, sheet_name="Best Trial").iloc[0]

    ascii_key   = str(best.get("ascii_key",   ""))
    role        = str(best.get("role",        ""))
    aims        = str(best.get("aims",        ""))
    description = str(best.get("description", ""))
    question    = str(best.get("question",    ""))

    print(f"\n{'='*55}")
    print(f"  BOP-SAP — Test Set Evaluation")
    print(f"  Model      : {args.model}")
    print(f"  Best trial : #{int(best['trial_number'])}")
    print(f"  ASCII key  : {ascii_key}")
    print(f"{'='*55}\n")

    # ── Load test data ─────────────────────────────────────────────────────────
    test_path = args.test_file or str(get_testing_xls())
    df_test   = pd.read_excel(test_path)
    print(f"Loaded {len(df_test)} test samples from {test_path}")

    # ── Build backend and system prompt ───────────────────────────────────────
    backend = build_backend(args.model)
    system  = assemble_system_prompt(role, aims, description, ascii_key)

    # ── Run inference ──────────────────────────────────────────────────────────
    records = []
    for _, row in df_test.iterrows():
        gene1    = str(row.get("starter",  row.get("Gene-A", ""))).upper()
        gene2    = str(row.get("receiver", row.get("Gene-B", ""))).upper()
        true_rel = row.get("relation_name", row.get("Ground truth", ""))
        user_q   = question.format(gene1=gene1, gene2=gene2)
        _, pred  = backend.query(system, user_q)
        records.append({
            "gene1":            gene1,
            "gene2":            gene2,
            "relation":         true_rel,
            "predict_relation": pred if pred else NO_INFO,
        })

    results_df = pd.DataFrame(records)
    metrics    = compute_all_metrics(results_df)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Results — {args.model}")
    print(f"{'─'*55}")
    print(f"  Macro F₁  : {metrics['macro_f1']:.4f}  "
          f"(paper: GPT-4/4o=0.80, LLaMA=0.66, Cohere=0.62)")
    print(f"  Micro F₁  : {metrics['micro_f1']:.4f}")
    print(f"  MCC       : {metrics['mcc']:.4f}")
    print(f"{'─'*55}")
    for rel in ["activation", "inhibition", "phosphorylation"]:
        print(f"  {rel.capitalize():16s} | "
              f"F1={metrics[f'{rel}_f1']:.4f}  "
              f"P={metrics[f'{rel}_precision']:.4f}  "
              f"R={metrics[f'{rel}_recall']:.4f}")
    print(f"{'─'*55}\n")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = args.output_path or str(EVAL_RESULTS_XLS)
    with pd.ExcelWriter(out_path) as writer:
        results_df.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
    print(f"✔ Evaluation results saved → {out_path}")


if __name__ == "__main__":
    main()

"""
src/scripts/run_optimization.py
================================
CLI entry point — Bayesian prompt optimisation.

Usage
-----
python src/scripts/run_optimization.py --model gpt-4o
python src/scripts/run_optimization.py --model llama3  --n_trials 50
python src/scripts/run_optimization.py --model cohere  --n_trials 50
python src/scripts/run_optimization.py --model gpt-3.5 --n_trials 50
python src/scripts/run_optimization.py --model gpt-4   --n_trials 50
"""

import argparse
from Configs.paths_config import ensure_dirs
from src.optimization.optimizer import run_optimization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BOP — Bayesian Prompt Optimisation")
    p.add_argument("--model", type=str, default="gpt-4o",
                   choices=["gpt-3.5", "gpt-4", "gpt-4o", "cohere", "llama3"])
    p.add_argument("--n_trials",       type=int, default=50)
    p.add_argument("--excel_path",     type=str, default=None)
    p.add_argument("--training_csv",   type=str, default=None)
    p.add_argument("--validation_csv", type=str, default=None)
    p.add_argument("--output_path",    type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    print(f"\n{'='*55}")
    print(f"  BOP-SAP — Bayesian Prompt Optimisation")
    print(f"  Model   : {args.model}")
    print(f"  Trials  : {args.n_trials}")
    print(f"{'='*55}\n")

    study = run_optimization(
        model_name     = args.model,
        excel_path     = args.excel_path,
        training_csv   = args.training_csv,
        validation_csv = args.validation_csv,
        output_path    = args.output_path,
        n_trials       = args.n_trials,
    )
    best = study.best_trial
    print(f"\n✅  Best trial #{best.number}")
    print(f"    Train F₁  : {best.value:.4f}")
    print(f"    Val   F₁  : {best.user_attrs.get('val_macro_f1', 'N/A')}")
    print(f"    ASCII key : {best.user_attrs.get('ascii_key', '')}")


if __name__ == "__main__":
    main()

"""
src/optimization/optimizer.py
==============================
BOP core: Bayesian prompt optimisation using Optuna TPE sampler.

Workflow:
  - Sample Role / Aims / Description / Question from Excel (2400 combinations)
  - Generate 15-char ASCII structural anchor per trial
  - System prompt = [ascii_key] + Role + Aims + Description + few-shot
  - Fitness = macro-F1 on training set
  - Validation macro-F1 tracked separately (never fed to Optuna)
  - TPE sampler, 10 startup random trials, 50 total
"""

import random
import string

import optuna
import pandas as pd

from Configs.config import (
    OPTUNA_N_TRIALS, OPTUNA_N_STARTUP_TRIALS,
    ASCII_KEY_LENGTH, PROMPT_SHEETS,
)
from Configs.paths_config import (
    get_prompt_parts_xls, get_training_csv,
    get_validation_csv, OPT_RESULTS_XLS,
)
from src.optimization.fitness import evaluate_prompt

optuna.logging.set_verbosity(optuna.logging.WARNING)


def random_ascii_key(length: int = ASCII_KEY_LENGTH) -> str:
    """Generate a random ASCII structural anchor."""
    chars = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choices(chars, k=length))


def load_prompt_parts(excel_path: str) -> dict[str, list[str]]:
    """Read all four SAP component lists from the Excel workbook."""
    parts: dict[str, list[str]] = {}
    for sheet, col in PROMPT_SHEETS.items():
        df = pd.read_excel(excel_path, sheet_name=sheet)
        parts[sheet] = df[col].dropna().tolist()
    return parts


def run_optimization(
    model_name:     str,
    excel_path:     str | None = None,
    training_csv:   str | None = None,
    validation_csv: str | None = None,
    output_path:    str | None = None,
    n_trials:       int = OPTUNA_N_TRIALS,
) -> optuna.Study:
    """
    Run BOP prompt optimisation for the specified model.

    Parameters
    ----------
    model_name     : 'gpt-3.5' | 'gpt-4' | 'gpt-4o' | 'cohere' | 'llama3'
    excel_path     : Path to prompt_parts.xlsx
    training_csv   : Path to training CSV
    validation_csv : Path to validation CSV (monitored — not optimised)
    output_path    : Where to save Excel results
    n_trials       : Optuna trials (default 50)
    """
    excel_path     = excel_path     or str(get_prompt_parts_xls())
    training_csv   = training_csv   or str(get_training_csv())
    validation_csv = validation_csv or str(get_validation_csv())
    output_path    = output_path    or str(OPT_RESULTS_XLS)

    parts = load_prompt_parts(excel_path)

    def objective(trial: optuna.Trial) -> float:
        role        = trial.suggest_categorical("role",        parts["roles"])
        aims        = trial.suggest_categorical("task",        parts["tasks"])
        description = trial.suggest_categorical("instruction", parts["general_instructions"])
        question    = trial.suggest_categorical("question",    parts["user_questions"])

        ascii_key = random_ascii_key()
        trial.set_user_attr("ascii_key", ascii_key)

        train_f1 = evaluate_prompt(
            model_name, role, aims, description, question, ascii_key, training_csv,
        )

        val_f1 = evaluate_prompt(
            model_name, role, aims, description, question, ascii_key, validation_csv,
        )
        trial.set_user_attr("val_macro_f1", val_f1)

        print(f"  Trial {trial.number:3d} | train={train_f1:.4f} "
              f"| val={val_f1:.4f} | key={ascii_key}")
        return train_f1

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(n_startup_trials=OPTUNA_N_STARTUP_TRIALS),
    )
    study.optimize(objective, n_trials=n_trials)
    _save_results(study, output_path, model_name)
    return study


def _save_results(study: optuna.Study, output_path: str, model_name: str) -> None:
    rows = [{
        "trial_number":  t.number,
        "model":         model_name,
        "train_macro_f1":t.value,
        "val_macro_f1":  t.user_attrs.get("val_macro_f1"),
        "ascii_key":     t.user_attrs.get("ascii_key", ""),
        "role":          t.params.get("role", ""),
        "aims":          t.params.get("task", ""),
        "description":   t.params.get("instruction", ""),
        "question":      t.params.get("question", ""),
    } for t in study.trials]

    best = study.best_trial
    best_row = {
        "trial_number":  best.number,
        "model":         model_name,
        "train_macro_f1":best.value,
        "val_macro_f1":  best.user_attrs.get("val_macro_f1"),
        "ascii_key":     best.user_attrs.get("ascii_key", ""),
        "role":          best.params.get("role", ""),
        "aims":          best.params.get("task", ""),
        "description":   best.params.get("instruction", ""),
        "question":      best.params.get("question", ""),
    }

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(rows).to_excel(writer,     sheet_name="All Trials", index=False)
        pd.DataFrame([best_row]).to_excel(writer,sheet_name="Best Trial", index=False)

    print(f"\n✔ Results saved → {output_path}")
    print(f"  Best #{best.number} | train={best.value:.4f} | key={best_row['ascii_key']}")

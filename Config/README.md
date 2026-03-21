# 🧬 BOP-SAP — Bayesian Optimization of Structural Anchor Prompting

**Paper:** *Bayesian Optimization of ASCII Structural Anchors for Improving Large Language Model Performance in Biomedical Knowledge Mining*
Muhammad Azam, Hasanain Aldihis, Shuai Zeng, Dong Xu — Under Review, 2026

[![GitHub](https://img.shields.io/badge/GitHub-BOP-blue)](https://github.com/Muh-aza/BOP)
[![Optuna](https://img.shields.io/badge/Optuna-TPE-green)](https://optuna.readthedocs.io)

---

## Overview

BOP-SAP is an automated, model-agnostic framework for optimising discrete prompt structures for biomedical gene–gene interaction classification. It uses **Bayesian Optimisation (Optuna TPE)** to search over structured SAP components — Role, Aims, Description, Question — augmented with a unique **15-character ASCII structural anchor** per trial.

**Supported models:** GPT-3.5 · GPT-4 · GPT-4o · Cohere Command-R+ · LLaMA-3.1-8B

---

## 📁 Project Structure

```
BOP/
├── Configs/
│   ├── __init__.py
│   ├── config.py              # master settings (hyperparams, API keys, UMAP)
│   ├── model_config.py        # per-model specs for all 5 LLMs
│   ├── paths_config.py        # all filesystem paths + env-var overrides
│   └── prompt_config.py       # SAP template (Role/Aims/Description/Question)
│
├── src/
│   ├── models/
│   │   ├── gpt_backend.py     # GPT-3.5 / GPT-4 / GPT-4o backend
│   │   ├── cohere_backend.py  # Cohere Command-R+ backend
│   │   ├── llama_backend.py   # LLaMA-3.1-8B: predict + hidden-state extract
│   │   └── model_factory.py   # factory + LLaMA singleton cache
│   │
│   ├── optimization/
│   │   ├── optimizer.py       # Optuna TPE loop + ASCII anchor + Excel save
│   │   └── fitness.py         # macro-F1 evaluator
│   │
│   ├── analysis/
│   │   ├── chi_analysis.py    # Calinski-Harabász Index per layer
│   │   ├── cosine_analysis.py # cosine similarity + gene trajectory
│   │   ├── umap_analysis.py   # UMAP grid + side-by-side comparison
│   │   └── plot_results.py    # convergence, cosine plots, trajectory plots
│   │
│   ├── utils/
│   │   ├── metrics.py         # F1/macro/micro/MCC/precision/recall
│   │   └── normaliser.py      # LLM output → canonical label + GPT fallback
│   │
│   └── scripts/
│       ├── run_optimization.py        # CLI: prompt optimisation
│       ├── evaluate_model.py          # CLI: test-set evaluation
│       └── run_embedding_analysis.py  # CLI: UMAP + cosine analysis
│
├── Data/dataset/Dataset/
│   ├── prompt_parts.xlsx      # SAP components (roles/tasks/instructions/questions)
│   ├── training.csv           # gene-pair training set (11 KEGG pathways)
│   ├── validation.csv         # held-out validation set
│   └── testing.xlsx           # held-out test set
│
├── Result/                    # all outputs (auto-created)
├── Model/                     # saved model artefacts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/Muh-aza/BOP.git
cd BOP
pip install -r requirements.txt
```

### 2. Set credentials

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
export HF_TOKEN="hf_..."
```

### 3. Run optimisation

```bash
python src/scripts/run_optimization.py --model gpt-4o   --n_trials 50
python src/scripts/run_optimization.py --model gpt-4    --n_trials 50
python src/scripts/run_optimization.py --model gpt-3.5  --n_trials 50
python src/scripts/run_optimization.py --model cohere   --n_trials 50
python src/scripts/run_optimization.py --model llama3   --n_trials 50
```

### 4. Evaluate on test set

```bash
python src/scripts/evaluate_model.py --model gpt-4o
```

### 5. Run embedding analysis

```bash
# Full pipeline:
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U"

# Anchor-only (faster):
python src/scripts/run_embedding_analysis.py --anchor_key "F3eI?%qt,NbnG8U" --skip_baseline
```

---

## 📊 Outputs

| File | Description |
|------|-------------|
| `Result/optimization_results.xlsx` | All Optuna trials + Best Trial |
| `Result/evaluation_results.xlsx` | Test predictions + metrics |
| `Result/CH_values.csv` | Calinski–Harabász index per layer |
| `Result/cosine_similarity.csv` | Per-class cosine similarity per layer |
| `Result/cosine_trajectory.xlsx` | Per gene-pair cosine trajectory |
| `Result/umap_grid_anchor.png` | UMAP grid — anchor prompt (600 dpi) |
| `Result/umap_grid_baseline.png` | UMAP grid — baseline prompt (600 dpi) |
| `Result/umap_comparison.png` | Side-by-side UMAP baseline vs anchor |
| `Result/cosine_classwise.png` | Per-class cosine similarity curve |
| `Result/cosine_average.png` | Average cosine similarity all gene pairs |
| `Result/cosine_trajectory.png` | Per-sample cosine trajectory |
| `Result/convergence_curve.png` | Train vs val F1 across iterations |

---

## 📖 Citation

```bibtex
@article{Azam2026BOPSAP,
  title   = {Bayesian Optimization of ASCII Structural Anchors for Improving
             Large Language Model Performance in Biomedical Knowledge Mining},
  author  = {Muhammad Azam and Hasanain Aldihis and Shuai Zeng and Dong Xu},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 📦 Dataset

Gene–gene interaction data from 11 KEGG signaling pathways.
Available at: [github.com/Muh-aza/BOP/tree/main/Data/dataset/data2](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

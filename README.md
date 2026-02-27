# BOP
Bayesian Optimization of Structural Anchor Prompting for Large Language Models

BOP-SAP (Bayesian Optimization – Structural Anchor Prompting) is a research framework for systematic prompt optimization in large language models (LLMs). The method formalizes prompt construction as a structured optimization problem over modular components and applies Bayesian Optimization (Tree-structured Parzen Estimator, TPE) to identify high-performing prompt configurations for biomedical relation extraction tasks.

The framework is developed for gene–gene interaction classification, including activation, inhibition, and phosphorylation prediction. Rather than relying on manual prompt engineering, BOP-SAP performs automated search across structured prompt components while keeping model parameters fixed, thereby isolating the structural contribution of prompt design to model performance.

<p align="center"> <img src="assets/figure1_framework.png" alt="Framework Overview" width="800"/> </p>
Overview

BOP-SAP consists of four principal stages:

Structured Prompt Design
Prompts are constructed from predefined components:

Role

Aim

Description

Question

Search Space Definition
The Cartesian product of prompt components defines a discrete optimization space.

Bayesian Optimization
Optuna’s Tree-structured Parzen Estimator proposes candidate configurations and iteratively refines the search distribution using validation performance as the objective.

Evaluation
Selected prompts are evaluated on held-out data using F1-score and Matthews Correlation Coefficient (MCC). Optional representation-level analyses examine structural consistency.

The LLM backbone remains frozen throughout optimization.

Supported Models

The framework is model-agnostic and supports:

GPT-4o

GPT-4

GPT-3.5

Cohere Command-R

Llama 3.x (local inference)

Additional models can be integrated via the llm_clients interface.

Installation

Clone the repository:

git clone https://github.com/yourusername/BOP-SAP.git
cd BOP-SAP

Install dependencies:

pip install -r requirements.txt

Optional (Conda environment):

conda env create -f environment.yml
conda activate bop-sap
Running Optimization

Execute Bayesian prompt search:

python src/scripts/run_optimization.py --model gpt-4o

Evaluate the best-performing configuration:

python src/scripts/evaluate_model.py --model gpt-4o

Optimization studies, logs, and selected prompts are stored under the results/ directory.

Reproducibility

The framework ensures experimental reproducibility through:

Fixed random seeds

Logged optimization trials

Stored Optuna study objects

Version-controlled configuration files

Deterministic evaluation procedures

Project Structure
BOP-SAP/
├── assets/                  # Figures
├── configs/                 # Model and optimization settings
├── data/                    # Datasets and splits
├── src/
│   ├── bop/                 # Core optimization framework
│   ├── llm_clients/         # Model interfaces
│   ├── evaluation/          # Metrics and scoring
│   └── analysis/            # Representation analysis
├── results/                 # Optimization outputs
├── docs/                    # Documentation
└── tests/                   # Unit tests
Citation
@article{Azam2026BOPSAP,
  title={Bayesian Optimization of String-Based Structural Anchor Prompting for Large Language Models},
  author={Muhammad Azam and Shuai Zeng and Hasanain Aldihis and Mihail Popescu and Toni Kazic and Duolin Wang and Dong Xu},
  year={2026},
  journal={Under Review}
}

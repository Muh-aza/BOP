# BOP
---

```text
BOP-SAP/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── setup.py
│
├── configs/                     # Experiment configurations
│   ├── models/
│   ├── optimization/
│   └── datasets/
│
├── data/                        # Dataset storage
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── assets/                      # Figures for README and papers
│   ├── framework.png
│   └── results.png
│
├── src/
│   ├── bop_sap/                 # Main library package
│   │   ├── __init__.py
│   │   │
│   │   ├── optimization/        # Bayesian optimization
│   │   │   ├── bayes_optimizer.py
│   │   │   └── search_space.py
│   │   │
│   │   ├── prompting/           # Prompt structure modules
│   │   │   ├── role.py
│   │   │   ├── aim.py
│   │   │   ├── description.py
│   │   │   └── question.py
│   │   │
│   │   ├── llm_clients/         # LLM API interfaces
│   │   │   ├── openai_client.py
│   │   │   ├── cohere_client.py
│   │   │   └── llama_client.py
│   │   │
│   │   ├── evaluation/          # Evaluation metrics
│   │   │   ├── metrics.py
│   │   │   └── scoring.py
│   │   │
│   │   ├── analysis/            # Representation analysis
│   │   │   ├── embedding_analysis.py
│   │   │   └── clustering.py
│   │   │
│   │   └── utils/
│   │       ├── logger.py
│   │       └── helpers.py
│
├── scripts/                     # Experiment scripts
│   ├── run_optimization.py
│   ├── evaluate_model.py
│   └── run_analysis.py
│
├── results/                     # Saved experiments
│   ├── trials/
│   └── best_prompts/
│
├── notebooks/                   # Research notebooks
│   ├── prompt_analysis.ipynb
│   └── representation_analysis.ipynb
│
├── docs/                        # Documentation
│   ├── methodology.md
│   └── architecture.md
│
└── tests/                       # Unit tests
    ├── test_optimizer.py
    └── test_prompts.py

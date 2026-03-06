# BOP
## 🚀 Bayesian Optimization of Structural Anchor Prompting for Large Language Models

BOP-SAP (Bayesian Optimization – Structural Anchor Prompting) is an open-source research framework for systematic prompt optimization in large language models (LLMs). The framework integrates structured anchor-based prompt design with Bayesian Optimization (Tree-structured Parzen Estimator, TPE) to enable automated, reproducible, and interpretable optimization of prompt configurations for biomedical gene–gene interaction classification tasks. Rather than relying on manual prompt engineering, BOP-SAP formulates prompt construction as a discrete optimization problem over modular components (Role, Aim, Description, Question). The method keeps model parameters fixed, isolating the structural contribution of prompt design to downstream performance.

<p align="center">
  <img src="https://github.com/Muh-aza/BOP/raw/main/Result/1.png" alt="BOP Framework Overview" width="600"/>
</p>

## 🌟 Features

- **Automated Bayesian Prompt Optimization:** Uses Optuna’s Tree-structured Parzen Estimator (TPE) to iteratively explore and optimize structured prompt components.

- **Structured Anchor Design:** Modular architecture based on Role, Aim, Description, and Question components.

- **Multi-LLM Support:** Compatible with GPT-4o, GPT-4, GPT-3.5, Cohere Command-R, and Llama 3.x.

- **Reproducible Optimization Pipeline:** Logged trials, stored study objects, and deterministic evaluation procedures.

- **Advanced Structural Analysis:** Optional hidden-state extraction, cosine similarity computation, and clustering-based representation analysis.

- **Open-source & Extensible:** Designed for integration of additional LLM backbones, datasets, and optimization strategies.

---

## ⚡ Quick Start
1. Clone the Repository
git clone https://github.com/yourusername/BOP-SAP.git
cd BOP-SAP
2. Install Dependencies
```bash
pip install -r requirements.txt
```
(Optional – Conda)
```bash
conda env create -f environment.yml
conda activate bop-sap
```
3. Run Bayesian Optimization
```bash
python src/scripts/run_optimization.py --model gpt-4o
```
4. Evaluate Optimized Prompt
```bash   
python src/scripts/evaluate_model.py --model gpt-4o

Results are stored in the results/ directory.
```
## 📁 Project Structure
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
## 📖 Citation

```bibtex
@article{Azam2026BOPSAP,
  title={Bayesian Optimization of String-Based Structural Anchor Prompting for Large Language Models},
  author={Muhammad Azam and Shuai Zeng and Hasanain Aldihis and Mihail Popescu and Toni Kazic and Duolin Wang and Dong Xu},
  journal={Under Review},
  year={2026}
}
```
  note={DOI: 10.1109/JBHI.2025.3631538}
}



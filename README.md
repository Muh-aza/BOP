# BOP
🚀 Bayesian Optimization of Structural Anchor Prompting for Large Language Models

BOP-SAP (Bayesian Optimization – Structural Anchor Prompting) is an open-source research framework that advances systematic prompt optimization for large language models (LLMs). By combining structured anchor-based prompt design with Bayesian Optimization (Tree-structured Parzen Estimator, TPE), BOP-SAP enables automated, reproducible, and interpretable optimization of prompt configurations for biomedical gene–gene interaction classification tasks.

BOP-SAP formalizes prompt construction as a discrete search problem across modular components (Role, Aim, Description, Question). The framework supports state-of-the-art LLMs including GPT-4o, GPT-4, Cohere Command-R, and Llama 3.x, providing a scalable and research-oriented alternative to manual prompt engineering approaches.

<p align="center"> <img src="assets/figure1_framework.png" alt="BOP-SAP Workflow" width="600"/> </p>
🌟 Features

Automated Bayesian Prompt Optimization: Uses Optuna’s Tree-structured Parzen Estimator (TPE) to iteratively explore and optimize structured prompt components.

Structured Anchor Design: Modular prompt architecture based on Role, Aim, Description, and Question components.

Multi-LLM Support: Compatible with GPT-4o, GPT-4, GPT-3.5, Cohere Command-R, and Llama 3.x.

Reproducible Optimization Pipeline: Logged trials, saved study objects, deterministic evaluation procedures.

Representation-Level Analysis: Optional hidden-state extraction, cosine similarity measurement, and clustering analysis.

Open-source & Extensible: Designed for integration of additional models, datasets, and optimization strategies.

⚡ Quick Start

Clone the Repository

git clone https://github.com/yourusername/BOP-SAP.git
cd BOP-SAP

Install Dependencies

pip install -r requirements.txt

Run Bayesian Optimization

python src/scripts/run_optimization.py --model gpt-4o

Evaluate Optimized Prompt

python src/scripts/evaluate_model.py --model gpt-4o
🎬 Demo

(Optional: add experiment visualization or workflow GIF here)

assets/demo.gif
🖥️ Usage

Configure model parameters in configs/models.yaml

Adjust search parameters in configs/optimization.yaml

Run optimization scripts under src/scripts/

Results are saved under the results/ directory

📁 Project Structure

Click here to view the full Project Structure →

📖 Citation
@article{Azam2026BOPSAP,
  title={Bayesian Optimization of String-Based Structural Anchor Prompting for Large Language Models},
  author={Muhammad Azam and Shuai Zeng and Hasanain Aldihis and Mihail Popescu and Toni Kazic and Duolin Wang and Dong Xu},
  journal={Under Review},
  year={2026}
}

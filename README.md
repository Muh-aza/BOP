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
### Bayesian Optimization

BOP-SAP performs prompt optimization using Bayesian Optimization implemented with the **Optuna** framework.

Optuna documentation can be found here:  [![Optuna Documentation](https://img.shields.io/badge/Optuna-Documentation-blue)](https://optuna.readthedocs.io/en/stable/)

The optimization process uses the **Tree-structured Parzen Estimator (TPE) sampler**, which models the distribution of promising and non-promising configurations to efficiently explore the prompt search space.

More information about the TPE sampler is available here: 
[![Optuna TPE Sampler](https://img.shields.io/badge/Optuna-TPE%20Sampler-green)](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

---
 1.**Clone the Repository**
```bash
git clone https://github.com/yourusername/BOP-SAP.git
cd BOP-SAP
```
 2. **Install Dependencies**
```bash
pip install -r requirements.txt
(Optional – Conda)
conda env create -f environment.yml
conda activate bop-sap
```
 3. **Run Bayesian Optimization**
```bash
python src/scripts/run_optimization.py --model gpt-4o
```
 4. **Evaluate Optimized Prompt**
```bash   
python src/scripts/evaluate_model.py --model gpt-4o
Results are stored in the results/ directory.
```
## 📁 Project Structure

[Click here to view the full Project Structure →](./Project1.md)

## 📊 Dataset

The BOP-SAP framework uses curated **biomedical signaling pathway datasets** containing gene–gene relationships extracted from KEGG pathways.  
KEGG pathways represent biological interaction networks describing molecular signaling and gene relationships in cells. :contentReference[oaicite:0]{index=0}

### Available Pathway Datasets

- **MAPK Signaling Pathway**   [![Dataset](https://img.shields.io/badge/Dataset-MAPK%20Signaling-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **EGFR Tyrosine Kinase Inhibitor Resistance**  
[![Dataset](https://img.shields.io/badge/Dataset-EGFR%20Resistance-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Endocrine Resistance**  
[![Dataset](https://img.shields.io/badge/Dataset-Endocrine%20Resistance-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Platinum Drug Resistance**  
[![Dataset](https://img.shields.io/badge/Dataset-Platinum%20Drug%20Resistance-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **ErbB Signaling Pathway**  
[![Dataset](https://img.shields.io/badge/Dataset-ErbB%20Pathway-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Ras Signaling Pathway**  
[![Dataset](https://img.shields.io/badge/Dataset-Ras%20Pathway-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Rap1 Signaling Pathway**  
[![Dataset](https://img.shields.io/badge/Dataset-Rap1%20Pathway-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **cGMP-PKG Signaling Pathway**  
[![Dataset](https://img.shields.io/badge/Dataset-cGMP--PKG%20Pathway-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **cAMP Signaling Pathway**  
[![Dataset](https://img.shields.io/badge/Dataset-cAMP%20Pathway-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Autophagy – Animal**  
[![Dataset](https://img.shields.io/badge/Dataset-Autophagy%20Animal-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

- **Endocytosis**  
[![Dataset](https://img.shields.io/badge/Dataset-Endocytosis-blue)](https://github.com/Muh-aza/BOP/tree/main/Data/dataset/data2)

### Gene Relationship Data

The datasets contain **gene–gene interaction relationships derived from KEGG signaling pathways**, which are used to evaluate the effectiveness of structural anchor prompt optimization in large language models.  
For example, the MAPK signaling pathway describes conserved molecular cascades involved in cell growth and differentiation. :contentReference[oaicite:1]{index=1}

## 📖 Citation

```bibtex
@article{Azam2026BOPSAP,
  title={Bayesian Optimization of String-Based Structural Anchor Prompting for Large Language Models},
  author={Muhammad Azam and Shuai Zeng and Hasanain Aldihis and Dong Xu},
  journal={Under Review},
  year={2026}
}
```

}



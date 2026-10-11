# Machine Learning

A personal repository for learning, practicing, and building projects in **Machine Learning and Artificial Intelligence**.

This repository documents my journey from ML fundamentals to practical, project-based learning.

## What You'll Find Here

- Machine learning concepts and notes
- Python implementations and experiments
- Data preprocessing and exploratory analysis
- Supervised and unsupervised learning
- Model training and evaluation
- Hands-on projects and Jupyter notebooks

## Current Project: Titanic Survival Prediction

The Titanic project explores a binary classification problem: predicting whether a passenger survived.

Two notebooks compare different workflow styles:

- **Without Pipeline:** preprocessing steps are handled manually.
- **Using Pipeline:** preprocessing and model steps are organized using Scikit-learn pipelines.

See the [Titanic project notes](projects/titanic/README.md).

## Tech Stack

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
machine-learning/
├── projects/
│   └── titanic/
│       └── README.md
├── titanic-without-using-pipeline.ipynb
├── titanic-using-pipeline.ipynb
├── train.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The Titanic notebooks and dataset currently live in the repository root. As the repository grows, projects can be moved into their own folders.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aisha453/machine-learning.git
cd machine-learning
```

### 2. Create and activate a virtual environment (recommended)

On Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open either Titanic notebook and run its cells. The dataset is currently at the repository root, so notebooks can load it with `pd.read_csv("train.csv")` when launched from that directory.

## Learning Roadmap

1. **Foundations:** Python, NumPy, Pandas, statistics, probability, and linear algebra.
2. **Core ML:** regression, classification, trees, ensemble methods, clustering, and dimensionality reduction.
3. **Model development:** preprocessing, feature engineering, train/test splits, cross-validation, tuning, and evaluation.
4. **Next steps:** deep learning, NLP, computer vision, and generative AI.

## Goal

Build a strong foundation in **AI/ML engineering** through consistent learning, experimentation, and practical projects.

---

*Learning. Building. Experimenting. Repeating.*

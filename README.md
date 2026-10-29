# Machine Learning

A personal repository for learning, practicing, and building projects in **Machine Learning and Artificial Intelligence**.

This repository documents the journey from ML fundamentals to practical, project-based learning.

## Current Project: Titanic Survival Prediction

The Titanic project explores binary classification: predicting whether a passenger survived.

- **[Without Pipeline](titanic-without-using-pipeline.ipynb):** manually organized preprocessing and modelling.
- **[Using Pipeline](titanic-using-pipeline.ipynb):** a complete Scikit-learn pipeline with column-based preprocessing, cross-validation, hyperparameter tuning, and model saving.
- **[Titanic project guide](projects/titanic/README.md):** setup instructions and concepts covered.
- **[Model comparison script](projects/titanic/compare_models.py):** compares Logistic Regression and Decision Tree with cross-validation and holdout metrics.

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aisha453/machine-learning.git
cd machine-learning
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the model comparison

```bash
python projects/titanic/compare_models.py
```

Or launch Jupyter and open either notebook:

```bash
jupyter notebook
```

The dataset `train.csv` is stored at the repository root. The notebooks expect to be launched from the repository root; the comparison script resolves the dataset path automatically.

## Learning Roadmap

1. **Foundations:** Python, NumPy, Pandas, statistics, probability, and linear algebra.
2. **Core ML:** regression, classification, trees, ensemble methods, clustering, and dimensionality reduction.
3. **Model development:** preprocessing, feature engineering, train/test splits, cross-validation, tuning, and evaluation.
4. **Next steps:** deep learning, NLP, computer vision, and generative AI.

## Goal

Build a strong foundation in **AI/ML engineering** through consistent learning, experimentation, and practical projects.

---

*Learning. Building. Experimenting. Repeating.*

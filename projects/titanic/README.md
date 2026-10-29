# Titanic Machine Learning Project

A binary classification project exploring whether a Titanic passenger survived. This project is designed to practice data preparation, model evaluation, and reproducible Scikit-learn workflows.

## Notebooks

### 1. Without Pipeline

[Open the original notebook](../../titanic-without-using-pipeline.ipynb)

Explore a manually organized preprocessing and modelling workflow.

### 2. Using Pipeline

[Open the pipeline notebook](../../titanic-using-pipeline.ipynb)

Practice column-based preprocessing with `ColumnTransformer`, model training with `Pipeline`, classification metrics, cross-validation, `GridSearchCV`, and saving the fitted model.

## Compare Two Models

Run the reusable comparison script from the repository root:

```bash
python projects/titanic/compare_models.py
```

It compares Logistic Regression and Decision Tree using the same stratified train/test split. It reports mean five-fold cross-validation accuracy on the training split and holdout metrics (accuracy, precision, recall, and F1). The holdout set is not used for model selection.

## Dataset and setup

The dataset is stored at the repository root as [train.csv](../../train.csv). From the repository root, install dependencies and run Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

The script locates `train.csv` relative to its own file, so it can be launched from any working directory.

## Concepts practiced

- Missing-value imputation and categorical encoding
- Column-specific preprocessing
- Train/test splitting with stratification
- Pipelines and leakage-safe cross-validation
- Accuracy, precision, recall, F1, and confusion matrices
- Hyperparameter tuning and model persistence
- Comparing a linear classifier with a tree-based classifier

## Notes

Metrics depend on the data and installed library versions. Run the notebook/script to generate fresh results; no fixed performance score is claimed here.

---

*Part of my ongoing Machine Learning learning journey.*

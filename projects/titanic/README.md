# Titanic Machine Learning Project

A machine learning classification project based on the Titanic passenger dataset.

## Project Overview

The goal is to explore a binary classification task: predicting whether a passenger survived the Titanic disaster. The notebooks compare manual preprocessing with a Scikit-learn pipeline-based workflow.

## Notebooks

### 1. Without Pipeline

[Open the titanic-without-using-pipeline notebook](../../titanic-without-using-pipeline.ipynb)

Explore preprocessing and modelling steps manually.

### 2. Using Pipeline

[Open the titanic-using-pipeline notebook](../../titanic-using-pipeline.ipynb)

This notebook uses `ColumnTransformer` and `Pipeline` to keep preprocessing and modelling together. It covers numerical and categorical imputation, one-hot encoding, scaling, model evaluation, cross-validation, hyperparameter tuning with `GridSearchCV`, and saving the fitted pipeline with Joblib.

## Dataset

The training dataset is stored at the repository root as [train.csv](../../train.csv). Run Jupyter from the repository root so the notebook can load it with `pd.read_csv("train.csv")`.

## Concepts Practiced

- Data inspection and feature/target separation
- Stratified train/test split
- Missing-value imputation
- Categorical encoding
- Column-wise preprocessing
- Scikit-learn pipelines
- Classification metrics and confusion matrix
- Cross-validation and hyperparameter tuning
- Saving a fitted pipeline for reuse

## Notes

The notebook does not commit fixed accuracy claims: rerun it in your environment to reproduce metrics with your installed library versions. The test set is kept separate from cross-validation and model selection.

---

*Part of my ongoing Machine Learning learning journey.*
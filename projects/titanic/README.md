# Titanic Machine Learning Project

A machine learning classification project based on the Titanic passenger dataset.

## Project Overview

The goal is to explore a binary classification task: predicting whether a passenger survived the Titanic disaster. The notebooks provide a way to compare a manually organized workflow with a pipeline-based workflow.

## Notebooks

### 1. Without Pipeline

[Open the titanic-without-using-pipeline notebook](../../titanic-without-using-pipeline.ipynb)

Explore preprocessing and modelling without packaging the workflow into a Scikit-learn pipeline.

### 2. Using Pipeline

[Open the titanic-using-pipeline notebook](../../titanic-using-pipeline.ipynb)

Explore how Scikit-learn pipelines can help organize preprocessing and modelling steps.

## Dataset

The training dataset is currently stored at the repository root as [train.csv](../../train.csv). Start Jupyter from the repository root so a relative read such as `pd.read_csv("train.csv")` can locate the file.

## Concepts to Explore

- Data inspection and preprocessing
- Missing-value handling
- Categorical feature encoding
- Train/test splitting
- Classification models
- Scikit-learn pipelines
- Model evaluation and comparison

The exact steps and results should be taken from the notebooks themselves; this README is a project guide, not a claim of a particular model score.

## Suggested Next Improvements

- Record baseline and final evaluation metrics
- Compare both workflows using the same split and evaluation metric
- Add a short conclusion about what the pipeline changes and why
- Keep notebook outputs and random seeds consistent for reproducibility

---

*Part of my ongoing Machine Learning learning journey.*
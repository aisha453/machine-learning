"""Compare baseline classifiers on the Titanic training dataset.

Run from any directory:
    python projects/titanic/compare_models.py

The script uses a stratified holdout set and cross-validation on the training split.
It does not use the holdout set for model selection.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[2] / "train.csv"


def build_pipeline(model, numeric_features, categorical_features):
    """Build preprocessing and classifier as a single leakage-safe pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Keep train.csv at the repository root."
        )

    data = pd.read_csv(DATA_PATH)
    if "Survived" not in data.columns:
        raise ValueError("The dataset must contain a 'Survived' target column.")

    target = data["Survived"]
    features = data.drop(
        columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
    )
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append(
            {
                "Model": name,
                "Mean CV Accuracy": cv_scores.mean(),
                "Test Accuracy": accuracy_score(y_test, predictions),
                "Test Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Test Recall": recall_score(y_test, predictions, zero_division=0),
                "Test F1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "Mean CV Accuracy", ascending=False
    )
    print("\nModel comparison (metrics are for this split):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

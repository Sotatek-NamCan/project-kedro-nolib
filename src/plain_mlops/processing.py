"""Data validation, splitting, modeling, and evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from pandera import DataFrameSchema
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_and_clean(
    customers: pd.DataFrame, *, schema: DataFrameSchema, drop_columns: Tuple[str, ...]
) -> pd.DataFrame:
    """Validate the raw data schema and drop configured columns."""
    df = schema.validate(customers)
    drop_cols = [col for col in drop_columns if col and col in df.columns]
    return df.drop(columns=drop_cols) if drop_cols else df


def split_features_label(
    df: pd.DataFrame, *, feature_cols: Tuple[str, ...], label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into feature matrix and target series."""
    X = df.loc[:, feature_cols].copy()
    y = df[label_col].astype(int).copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _resolve_model_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the selected model definition."""
    if not model_cfg:
        raise ValueError("Model configuration section is missing.")

    if "type" in model_cfg:
        return {
            "type": model_cfg["type"],
            "hyperparameters": model_cfg.get("hyperparameters", {}),
        }

    options = model_cfg.get("options") or {}
    if not isinstance(options, dict) or not options:
        raise ValueError("Model configuration must include at least one option.")

    selected = model_cfg.get("selected")
    if selected:
        try:
            chosen = options[selected]
        except KeyError as exc:
            available = ", ".join(sorted(options))
            raise KeyError(
                f"Selected model '{selected}' not found. Available: {available}"
            ) from exc
    elif len(options) == 1:
        chosen = next(iter(options.values()))
    else:
        raise ValueError("Multiple model options provided but none selected.")

    if "type" not in chosen:
        raise ValueError("Each model option must define a 'type'.")

    return {
        "type": chosen["type"],
        "hyperparameters": chosen.get("hyperparameters", {}),
    }


def _create_estimator(model_def: Dict[str, Any]):
    model_type = model_def["type"]
    hyperparams = model_def.get("hyperparameters") or {}

    if model_type == "logreg":
        return LogisticRegression(**hyperparams)
    if model_type == "rf":
        return RandomForestClassifier(**hyperparams)

    raise ValueError(f"Unsupported model type: {model_type}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    numeric_cols: Tuple[str, ...],
    categorical_cols: Tuple[str, ...],
    model_cfg: Dict[str, Any],
):
    """Train a churn model using the configured estimator and preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_cols),
            ),
        ],
        remainder="drop",
    )

    estimator = _create_estimator(_resolve_model_config(model_cfg))
    pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Return evaluation metrics for the fitted model."""
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def run_training_workflow(
    df: pd.DataFrame,
    *,
    schema: DataFrameSchema,
    drop_columns: Tuple[str, ...],
    feature_cols: Tuple[str, ...],
    label_col: str,
    numeric_cols: Tuple[str, ...],
    categorical_cols: Tuple[str, ...],
    splitter_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Tuple[Pipeline, Dict[str, float], DatasetSplits]:
    """Execute the full validation, split, train, and evaluate workflow."""
    cleaned = validate_and_clean(df, schema=schema, drop_columns=drop_columns)
    X, y = split_features_label(
        cleaned, feature_cols=feature_cols, label_col=label_col
    )
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=splitter_cfg["test_size"],
        random_state=splitter_cfg["random_state"],
    )
    model = train_model(
        X_train,
        y_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
    )
    metrics = evaluate(model, X_test, y_test)
    splits = DatasetSplits(X_train, X_test, y_train, y_test)
    return model, metrics, splits


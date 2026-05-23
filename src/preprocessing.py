"""Preprocessing functions for model training and inference."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the raw dataframe into features and binary target.

    The notebook used pd.get_dummies(drop_first=True), producing the target column
    `IncomeClass_ >50K`. The production pipeline keeps the raw target column and
    converts ` >50K` to 1 and all other values to 0.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing transformer.

    This is the production-style equivalent of the notebook's:
    `pd.get_dummies(df, drop_first=True)`.
    """
    categorical_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_encoder, CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

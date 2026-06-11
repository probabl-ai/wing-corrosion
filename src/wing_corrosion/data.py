"""Data loading and X-marker wiring.

Owns: how raw data is materialized into `(X, y)`, and how structural
metadata (groups, time ordering, ...) is attached at the X marker via
`split_kwargs`. Pipeline mechanics live in `build-ml-pipeline`; data
paths are decided by the caller — this module does not invent a
`data/` directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_training_data(data_dir: str | Path):
    """Load training environment and corrosion data.

    Returns
    -------
    X : pd.DataFrame
        Monthly environmental features with aircraft_id and year_month
    y : pd.Series
        Binary target (1 = corrosion observed, 0 = 24 months before)
    """
    data_dir = Path(data_dir)

    # Load environment data
    env_train = pd.read_csv(data_dir / "environment_training.csv")

    # Load corrosion observations
    corr_train = pd.read_csv(data_dir / "corrosions_training.csv")

    # Build target: 1 at observation_date, 0 at 24 months before
    corr_train["observation_date"] = pd.to_datetime(corr_train["observation_date"])

    # Create positive samples (corrosion observed)
    positive = corr_train.copy()
    positive["year_month"] = positive["observation_date"].dt.to_period("M").astype(str)
    positive["corrosion_risk"] = 1

    # Create negative samples (24 months before observation)
    negative = corr_train.copy()
    negative["observation_date"] = negative["observation_date"] - pd.DateOffset(
        months=24
    )
    negative["year_month"] = negative["observation_date"].dt.to_period("M").astype(str)
    negative["corrosion_risk"] = 0

    # Combine positive and negative samples
    target_df = pd.concat([positive, negative])[
        ["aircraft_id", "year_month", "corrosion_risk"]
    ]

    # Merge with environment data
    data = env_train.merge(target_df, on=["aircraft_id", "year_month"], how="inner")

    # Separate features and target
    feature_cols = [
        col
        for col in data.columns
        if col not in ["aircraft_id", "year_month", "corrosion_risk"]
    ]
    X = data[["aircraft_id", "year_month"] + feature_cols]
    y = data["corrosion_risk"]

    return X, y


def load_test_data(data_dir: str | Path):
    """Load test environment data.

    Returns
    -------
    X_test : pd.DataFrame
        Monthly environmental features with aircraft_id and year_month
    """
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "environment_test.csv")


# Made with Bob

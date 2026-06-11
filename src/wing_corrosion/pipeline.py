"""Learner declaration.

Owns: the function that builds and returns the (unfit) learner —
typically a `SkrubLearner` produced from a skrub DataOps graph that
composes the steps in `data.py` and `features.py` with the chosen
estimator. Fitting, evaluation, and persistence happen elsewhere.
See `build-ml-pipeline` for the declarative mechanics.
"""

from __future__ import annotations

from pathlib import Path

import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

from wing_corrosion.data import load_training_data


def build_learner(data_dir_preview: str | Path | None = None):
    """Return the unfit learner for the experiment scripts to consume.

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Preview value for the source-bound `skrub.var("data_dir", ...)`
        root. Pass an absolute path (e.g. `<pkg>.PROJECT_ROOT / "data"`)
        when iterating interactively so `learner.skb.preview()` works.
        Leave as `None` for fit / cross-validate runs — the env-dict
        passed to `skore.evaluate(..., data={"data_dir": ...})`
        supplies the binding regardless.
    
    Returns
    -------
    learner : SkrubLearner
        Unfit learner ready for cross-validation via skore.evaluate.
    """
    # Layer 1: Source binding
    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )
    
    # Load data - returns a tuple (X, y)
    data = data_dir.skb.apply_func(load_training_data)
    
    # Layer 2: Extract X and y, then mark
    # Drop metadata columns (aircraft_id, year_month, month_start_date)
    X_features = data.skb.apply_func(
        lambda xy: xy[0].drop(
            columns=["aircraft_id", "year_month", "month_start_date"]
        )
    ).skb.mark_as_X()
    y_target = data.skb.apply_func(lambda xy: xy[1]).skb.mark_as_y()
    
    # Layer 3: Estimator
    # Use HistGradientBoostingClassifier (handles missing values natively)
    predictions = X_features.skb.apply(
        HistGradientBoostingClassifier(random_state=42),
        y=y_target
    )
    
    return predictions.skb.make_learner()

# Made with Bob

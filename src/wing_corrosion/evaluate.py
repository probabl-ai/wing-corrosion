"""Inputs to `skore.evaluate`.

This module is **strictly limited** to objects consumed by
`skore.evaluate(...)` calls in the experiment scripts:

- `splitter` — the cross-validator (chosen per
  `evaluate-ml-pipeline`); fed straight into
  `skore.evaluate(..., splitter=splitter)`,
- optional metric overrides, only when the user has explicitly
  asked for them (skore picks task-appropriate defaults otherwise).

It must NOT call `skore.evaluate`, open a `skore.Project`, or
persist anything. Those steps belong in the experiment script.
"""

from __future__ import annotations

from sklearn.model_selection import KFold

# 5-fold cross-validation with shuffling for IID data
# Brier score (default for binary classification) measures calibration quality
splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# Made with Bob

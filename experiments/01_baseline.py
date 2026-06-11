# %% [markdown]
# # Experiment 01: Baseline
#
# **Date:** 2026-06-11
# **Goal:** Establish baseline for predicting aircraft corrosion risk
#           from environmental features
# **Result:** (to be filled after run)

# %%
import skore

from wing_corrosion import PROJECT_ROOT
from wing_corrosion.evaluate import splitter
from wing_corrosion.pipeline import build_learner

# %% [markdown]
# ## Paths
#
# `PROJECT_ROOT` comes from the package's `__init__.py` and resolves
# from `__file__` — independent of the current working directory.

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project
#
# Local mode: reports persist to disk under `reports/`.
# No account or login required.

# %%
project = skore.Project(
    name="wing-corrosion",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

# %% [markdown]
# ## Data and learner
#
# `data_dir_preview=DATA_DIR` makes `learner.skb.preview()` work; it
# does not affect what `skore.evaluate` actually fits on (that comes
# from `data=` below).

# %%
learner = build_learner(data_dir_preview=DATA_DIR)

# %% [markdown]
# ## Evaluate
#
# Cross-validator (5-fold KFold) is imported from `wing_corrosion.evaluate`.
# The experiment script does not redefine it.
# `SkrubLearner.fit` takes a single environment dict, so we pass the
# bindings via `data=`.

# %%
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)}, splitter=splitter)

# %% [markdown]
# ## Persist
#
# Key = file stem. Reusing this key in a future run overwrites the
# stored report — fork into a new experiment file if you want both.

# %%
project.put("01_baseline", report)

# %%
# Display summary instead of full report to avoid rich rendering issues
print(f"Report stored as '01_baseline'")
print(f"Cross-validation completed with {splitter.n_splits} folds")

# Made with Bob

# %% [markdown]
# # Wing Corrosion — v1 baseline (instantaneous env, HistGradientBoosting)
#
# Self-contained Kaggle notebook source (jupytext py:percent). Paired notebook
# (`wing-corrosion-baseline.ipynb`) is what gets pushed to Kaggle; the `.ipynb`
# is gitignored per the repo's jupytext-source-of-truth policy.
#
# Predict an aircraft's monthly `corrosion_risk` from its environmental exposure
# history. Target convention (Airbus): the observation month is positive (1) and
# the month exactly 24 months earlier is negative (0). Metric: Brier score on the
# reference points (constant 0.5 -> 0.25 is the floor).
#
# **Public LB Brier: 0.27171**

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CANDIDATES = [
    Path("/kaggle/input/haks-airbus-x-ibm-x-aws-2026"),
    Path("../../data"),
    Path("data"),
]
DATA_DIR = next((p for p in CANDIDATES if (p / "environment_test.csv").exists()), None)
if DATA_DIR is None:
    hits = list(Path("/kaggle/input").glob("**/environment_test.csv"))
    DATA_DIR = hits[0].parent if hits else Path(".")
print("DATA_DIR =", DATA_DIR)

ID, YM = "aircraft_id", "year_month"
META = [ID, YM, "month_start_date"]

# %% [markdown]
# ## Build the (1 / 0) target and merge with environmental features

# %%
env_train = pd.read_csv(DATA_DIR / "environment_training.csv")
corr_train = pd.read_csv(DATA_DIR / "corrosions_training.csv")
corr_train["observation_date"] = pd.to_datetime(corr_train["observation_date"])

pos = corr_train.copy()
pos[YM] = pos["observation_date"].dt.to_period("M").astype(str)
pos["corrosion_risk"] = 1
neg = corr_train.copy()
neg["observation_date"] = neg["observation_date"] - pd.DateOffset(months=24)
neg[YM] = neg["observation_date"].dt.to_period("M").astype(str)
neg["corrosion_risk"] = 0
target_df = pd.concat([pos, neg])[[ID, YM, "corrosion_risk"]]

data = env_train.merge(target_df, on=[ID, YM], how="inner")
feature_cols = [c for c in data.columns if c not in META + ["corrosion_risk"]]
X = data[feature_cols]
y = data["corrosion_risk"]
print(f"training pairs: {len(data)}  positives={int(y.sum())}  features={len(feature_cols)}")

# %% [markdown]
# ## Train (HistGradientBoosting handles missing values natively)

# %%
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X, y)

# %% [markdown]
# ## Predict on the full test history and write the submission

# %%
env_test = pd.read_csv(DATA_DIR / "environment_test.csv")
proba = model.predict_proba(env_test[feature_cols])[:, 1]
submission = pd.DataFrame(
    {"id": env_test[ID] + "_" + env_test[YM], "corrosion_risk": proba}
)
submission.to_csv("submission.csv", index=False)
print(submission.shape)

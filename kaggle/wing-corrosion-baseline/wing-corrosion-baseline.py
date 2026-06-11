# %% [markdown]
# # Wing Corrosion — v2 (time + cumulative exposure, calibrated)
#
# Self-contained Kaggle notebook source (jupytext py:percent). Paired notebook
# (`wing-corrosion-baseline.ipynb`) is what gets pushed to Kaggle; the `.ipynb`
# is gitignored per the repo's jupytext-source-of-truth policy.
#
# **Key idea over v1:** the positive (observation month) and negative (24 months
# earlier) are the *same aircraft at the same airport*, so the *instantaneous*
# monthly environment barely separates them. Corrosion is progressive, so the
# real signal is **time** and **cumulative exposure**:
#
# - `age_months` — months since the aircraft's first record (delivery proxy);
# - `mean_<feature>` — running cumulative *mean* of each variable (a level, not a
#   total -> robust to the train/test shift where test aircraft are older).
#
# Classifier is isotonic-calibrated (Brier rewards calibrated probabilities).
#
# Local 5-fold GroupKFold Brier (grouped by aircraft): 0.142 vs 0.207 for v1.
#
# **Public LB Brier: 0.23211**

# %%
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
# ## Feature engineering
#
# Computed per aircraft over its full monthly history, in time order. Cumulative
# features are strictly within-aircraft and causal, so applying this to the test
# set independently introduces no leakage.

# %%
def add_features(df, feat_cols):
    df = df.sort_values([ID, YM]).copy()
    ymp = pd.PeriodIndex(df[YM], freq="M")
    firstp = pd.PeriodIndex(df.groupby(ID)[YM].transform("min"), freq="M")
    df["age_months"] = (ymp - firstp).map(lambda x: x.n)
    g = df.groupby(ID)
    for c in feat_cols:
        df[f"mean_{c}"] = g[c].cumsum() / (df["age_months"] + 1)
    return df


# %% [markdown]
# ## Build the (1 / 0) target and the training matrix

# %%
env_train = pd.read_csv(DATA_DIR / "environment_training.csv")
corr_train = pd.read_csv(DATA_DIR / "corrosions_training.csv")
corr_train["observation_date"] = pd.to_datetime(corr_train["observation_date"])

raw_feats = [c for c in env_train.columns if c not in META]
env_train = add_features(env_train, raw_feats)

pos = corr_train.copy()
pos[YM] = pos["observation_date"].dt.to_period("M").astype(str)
pos["corrosion_risk"] = 1
neg = corr_train.copy()
neg["observation_date"] = neg["observation_date"] - pd.DateOffset(months=24)
neg[YM] = neg["observation_date"].dt.to_period("M").astype(str)
neg["corrosion_risk"] = 0
target_df = pd.concat([pos, neg])[[ID, YM, "corrosion_risk"]]

data = env_train.merge(target_df, on=[ID, YM], how="inner")
feature_cols = raw_feats + ["age_months"] + [f"mean_{c}" for c in raw_feats]
X = data[feature_cols]
y = data["corrosion_risk"]
print(f"training pairs: {len(data)}  positives={int(y.sum())}  features={len(feature_cols)}")

# %% [markdown]
# ## Train: isotonic-calibrated HistGradientBoosting

# %%
model = CalibratedClassifierCV(
    HistGradientBoostingClassifier(random_state=42),
    method="isotonic",
    cv=3,
)
model.fit(X, y)

# %% [markdown]
# ## Predict on the full test history and write the submission

# %%
env_test = pd.read_csv(DATA_DIR / "environment_test.csv")
env_test = add_features(env_test, raw_feats)
proba = model.predict_proba(env_test[feature_cols])[:, 1]
submission = pd.DataFrame(
    {"id": env_test[ID] + "_" + env_test[YM], "corrosion_risk": proba}
)
submission.to_csv("submission.csv", index=False)
print(submission.shape)

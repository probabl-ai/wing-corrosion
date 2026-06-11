# %% [markdown]
# # Wing Corrosion — v3 (true age-since-delivery + cumulative exposure, calibrated)
#
# Self-contained Kaggle notebook source (jupytext py:percent). Paired notebook
# (`wing-corrosion-baseline.ipynb`) is what gets pushed to Kaggle; the `.ipynb`
# is gitignored per the repo's jupytext-source-of-truth policy.
#
# Corrosion is progressive, so the dominant signal is **age since delivery** plus
# **cumulative environmental exposure**. Age-at-first-observed-corrosion is sharply
# peaked (~5-6 years); the negative reference point is exactly 24 months earlier.
#
# **Key fix over v2:** v2 approximated age as *months since the first environmental
# record*, but that matches true delivery only ~43% of the time in training. Here:
#
# - **Training** uses the *true* delivery date (from corrosions_training);
# - **Test** has no delivery date, but the test fleet was all delivered in 2014 and
#   their histories start at delivery, so we use the first record as the proxy.
#
# This makes age_months mean the same real quantity in both sets, which transfers
# better to the older 2014 test fleet.
#
# **Public LB Brier: 0.21718**

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
# `delivery` is a per-aircraft Period('M'). Cumulative means are within-aircraft
# and causal.

# %%
def add_features(df, feat_cols, delivery):
    """delivery: Series indexed by aircraft_id giving the delivery Period('M')."""
    df = df.sort_values([ID, YM]).copy()
    ymp = pd.PeriodIndex(df[YM], freq="M")
    dvals = pd.PeriodIndex(delivery.reindex(df[ID]).values, freq="M")
    df["age_months"] = [(a - d).n for a, d in zip(ymp, dvals)]
    g = df.groupby(ID)
    denom = df["age_months"].clip(lower=0) + 1
    for c in feat_cols:
        df[f"mean_{c}"] = g[c].cumsum() / denom
    return df


# %% [markdown]
# ## Build the (1 / 0) target and the training matrix (true delivery age)

# %%
env_train = pd.read_csv(DATA_DIR / "environment_training.csv")
corr_train = pd.read_csv(DATA_DIR / "corrosions_training.csv")
corr_train["observation_date"] = pd.to_datetime(corr_train["observation_date"])

# True delivery period per training aircraft
deliv_train = pd.PeriodIndex(
    pd.to_datetime(
        dict(
            year=corr_train["aircraft_delivery_year"],
            month=corr_train["aircraft_delivery_month"],
            day=1,
        )
    ),
    freq="M",
)
delivery_train = pd.Series(deliv_train, index=corr_train[ID])
delivery_train = delivery_train[~delivery_train.index.duplicated()]

raw_feats = [c for c in env_train.columns if c not in META]
env_train = add_features(env_train, raw_feats, delivery_train)

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
# ## Predict on test (delivery proxy = first environmental record)

# %%
env_test = pd.read_csv(DATA_DIR / "environment_test.csv")

# Test has no delivery date; the 2014 fleet's history starts at delivery, so use
# each aircraft's first recorded month as the delivery proxy.
first_month = env_test.groupby(ID)[YM].min()
delivery_test = pd.Series(
    pd.PeriodIndex(first_month, freq="M"), index=first_month.index
)

env_test = add_features(env_test, raw_feats, delivery_test)
proba = model.predict_proba(env_test[feature_cols])[:, 1]
submission = pd.DataFrame(
    {"id": env_test[ID] + "_" + env_test[YM], "corrosion_risk": proba}
)
submission.to_csv("submission.csv", index=False)
print(submission.shape)

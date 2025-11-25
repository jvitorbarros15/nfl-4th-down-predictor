import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from xgboost import XGBClassifier


sns.set_theme(style="whitegrid")

# --------------------
# Streamlit config
# --------------------
st.set_page_config(page_title="NFL 4th Down Predictor", layout="wide")
st.title("🏈 NFL 4th Down Conversion Explorer")


# --------------------
# Data loading
# --------------------
st.sidebar.header("Data")

data_choice = st.sidebar.radio(
    "How do you want to load the data?",
    ["Local file path", "Upload file"],
    index=0
)

@st.cache_data
def load_df_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Please use a .parquet or .csv file")

df = None

if data_choice == "Local file path":
    default_path = "nfl_pbp_2013_2024.parquet"
    data_path = st.sidebar.text_input("File path", value=default_path)
    if st.sidebar.button("Load data"):
        try:
            df = load_df_from_path(data_path)
            st.success(f"Loaded {df.shape[0]:,} rows from {data_path}")
        except Exception as e:
            st.error(str(e))
else:
    uploaded = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.success(f"Loaded {df.shape[0]:,} rows from upload")
        except Exception as e:
            st.error(str(e))

if df is None:
    st.info("Load a dataset in the sidebar to start.")
    st.stop()


# --------------------
# Basic schema checks
# --------------------
needed_cols = [
    "GameId", "Quarter", "Minute", "Second",
    "OffenseTeam", "DefenseTeam",
    "Down", "ToGo", "YardLine",
    "IsRush", "IsPass", "IsNoPlay",
    "IsTouchdown", "IsInterception", "IsSack", "IsIncomplete"
]

missing = [c for c in needed_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns detected, filling with zeros: {missing}")
    for c in missing:
        df[c] = 0

for c in ["Down", "ToGo", "YardLine", "Quarter", "Minute", "Second",
          "IsRush", "IsPass", "IsNoPlay", "IsTouchdown",
          "IsInterception", "IsSack", "IsIncomplete"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# chronological sort (NFL clock counts down)
df = df.sort_values(
    ["GameId", "Quarter", "Minute", "Second"],
    ascending=[True, True, False, False]
).reset_index(drop=True)


# --------------------
# Label 4th down attempts and success
# --------------------
is_4th = df["Down"].eq(4)
is_attempt = ((df["IsRush"] == 1) | (df["IsPass"] == 1)) & (df["IsNoPlay"] == 0)
mask_4th_attempt = is_4th & is_attempt

g = df.groupby("GameId", group_keys=False)
df["NextDown"] = g["Down"].shift(-1)
df["NextOffenseTeam"] = g["OffenseTeam"].shift(-1)

success_simple = (
    (df["OffenseTeam"] == df["NextOffenseTeam"]) &
    (df["NextDown"] == 1)
) | (df["IsTouchdown"] == 1)

df["FourthDownSuccess_simple"] = (mask_4th_attempt & success_simple).astype(int)

attempts = df.loc[mask_4th_attempt].copy()
if attempts.empty:
    st.error("No 4th down attempts found after filtering. Check your dataset.")
    st.stop()

st.write(f"Total 4th down go for it attempts found: **{attempts.shape[0]:,}**")


# --------------------
# Simple feature engineering
# --------------------
attempts["ShortYards"] = (attempts["ToGo"] <= 2).astype(int)
attempts["InOwnHalf"] = (attempts["YardLine"] < 50).astype(int)
attempts["RedZone"] = (attempts["YardLine"] >= 80).astype(int)
attempts["LateGame"] = ((attempts["Quarter"] >= 4) & (attempts["Minute"] <= 5)).astype(int)
attempts["Clock"] = (attempts["Quarter"] * 900 - (attempts["Minute"] * 60 + attempts["Second"])).astype(int)

# filter by yards to go
st.sidebar.header("Filter")
max_togo = int(attempts["ToGo"].max())
togo_cap = st.sidebar.slider("Max To Go", 1, max(1, max_togo), min(15, max_togo))

attempts_filt = attempts.query("ToGo <= @togo_cap").copy()
if attempts_filt.shape[0] < 50:
    st.warning("Very few attempts after ToGo filter. Consider increasing the cap.")
    attempts_filt = attempts.copy()

features = [
    "ToGo", "YardLine", "IsRush", "IsPass",
    "ShortYards", "InOwnHalf", "RedZone", "LateGame", "Clock"
]
features = [f for f in features if f in attempts_filt.columns]

X = attempts_filt[features].fillna(0).astype(float)
y = attempts_filt["FourthDownSuccess_simple"].astype(int)
groups = attempts_filt["GameId"]


# --------------------
# Train and test split with group shuffle
# --------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# --------------------
# XGBoost model
# --------------------
pos_w = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=pos_w,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("ROC AUC", f"{auc:.3f}")
c3.metric("Brier Score", f"{brier:.3f}")


# --------------------
# Visuals
# --------------------
st.header("Visuals")

# 1. Pass vs Run conversion rate
palette_pr = {"Pass": "#FF6B6B", "Run": "#4ECDC4", "Other": "gray"}

attempts_filt["PlayTypeSimple"] = np.where(
    attempts_filt["IsPass"] == 1, "Pass",
    np.where(attempts_filt["IsRush"] == 1, "Run", "Other")
)

type_rates = (
    attempts_filt.groupby("PlayTypeSimple")["FourthDownSuccess_simple"]
    .mean()
    .sort_values(ascending=False)
)

df_plot = type_rates.reset_index()
df_plot.columns = ["PlayType", "Rate"]

fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.barplot(
    data=df_plot,
    x="PlayType",
    y="Rate",
    hue="PlayType",
    palette=palette_pr,
    dodge=False,
    ax=ax1
)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Play Type")
ax1.set_ylabel("Conversion Rate")
ax1.set_title("4th Down Conversion: Pass vs Run")
leg = ax1.get_legend()
if leg:
    leg.remove()
st.pyplot(fig1)


# 2. Predicted probability vs ToGo, colored by play type
plot_df = X_test[["ToGo"]].copy()
plot_df["prob"] = y_prob
plot_df["Play"] = np.where(
    X_test.get("IsPass", 0) == 1, "Pass",
    np.where(X_test.get("IsRush", 0) == 1, "Run", "Other")
)

plot_df = plot_df.query("ToGo <= 15").copy()
if len(plot_df) > 20000:
    plot_df = plot_df.sample(20000, random_state=42)

fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=plot_df,
    x="ToGo",
    y="prob",
    hue="Play",
    palette=palette_pr,
    alpha=0.35,
    s=25,
    edgecolor=None,
    ax=ax2
)
ax2.set_xlim(-0.2, 15.5)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Yards To Go")
ax2.set_ylabel("Predicted Conversion Probability")
ax2.set_title("4th Down Conversion Probability vs Yards To Go")
ax2.legend(title="", loc="upper right")
st.pyplot(fig2)


# 3. Team aggressiveness vs execution
team_rates = (
    attempts_filt.groupby("OffenseTeam")["FourthDownSuccess_simple"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "conv_rate", "count": "attempts"})
    .sort_values("conv_rate", ascending=False)
)

team_playstyle = team_rates.reset_index()

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=team_playstyle,
    x="attempts",
    y="conv_rate",
    ax=ax3
)
for _, row in team_playstyle.iterrows():
    ax3.text(
        row["attempts"] + 0.3,
        row["conv_rate"],
        row["OffenseTeam"],
        fontsize=8
    )

ax3.set_xlabel("Total 4th Down Attempts")
ax3.set_ylabel("Conversion Rate")
ax3.set_title("Team 4th Down Strategy: Aggressiveness vs Execution")
st.pyplot(fig3)


# --------------------
# Data preview
# --------------------
with st.expander("Preview 4th down attempts data"):
    st.dataframe(
        attempts_filt[
            [
                "GameId", "OffenseTeam", "DefenseTeam", "Down", "ToGo", "YardLine",
                "IsRush", "IsPass", "ShortYards", "InOwnHalf",
                "RedZone", "LateGame", "Clock", "FourthDownSuccess_simple"
            ]
        ].head(50)
    )
st.write("Data loaded and processed successfully.")
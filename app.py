import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from xgboost import XGBClassifier

# --------------------
# Streamlit setup
# --------------------

st.set_page_config(page_title="NFL 4th Down Predictor", layout="wide")
sns.set_theme(style="whitegrid")

st.markdown(
    "<h1 style='text-align:center;'>🏈 NFL 4th Down Conversion Explorer</h1>",
    unsafe_allow_html=True,
)
st.write("Interactive prototype that models and visualizes 4th down decisions.")


# --------------------
# Data loading helpers
# --------------------

@st.cache_data
def load_df_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Please use a .parquet or .csv file")


@st.cache_data
def load_df_from_upload(content: bytes, name: str):
    from io import BytesIO
    if name.endswith(".parquet"):
        return pd.read_parquet(BytesIO(content))
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(content))
    raise ValueError("Please use a .parquet or .csv file")


# --------------------
# Sidebar: data source
# --------------------

st.sidebar.header("Data source")

data_choice = st.sidebar.radio(
    "Load data from",
    ["Local file path", "Upload file"],
    index=0,
)

df = None

if data_choice == "Local file path":
    default_path = "nfl_pbp_2013_2024.parquet"
    data_path = st.sidebar.text_input("File path", value=default_path)
    if data_path:
        try:
            df = load_df_from_path(data_path)
        except Exception as e:
            st.sidebar.error(str(e))
else:
    uploaded = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    if uploaded is not None:
        try:
            df = load_df_from_upload(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.sidebar.error(str(e))

if df is None:
    st.info("Set a valid file path or upload a file in the sidebar to begin.")
    st.stop()

st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")

# --------------------
# Basic prep
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
    st.warning(f"Missing columns filled with zeros: {missing}")
    for c in missing:
        df[c] = 0

for c in [
    "Down", "ToGo", "YardLine", "Quarter", "Minute", "Second",
    "IsRush", "IsPass", "IsNoPlay",
    "IsTouchdown", "IsInterception", "IsSack", "IsIncomplete"
]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

df = df.sort_values(
    ["GameId", "Quarter", "Minute", "Second"],
    ascending=[True, True, False, False]
).reset_index(drop=True)

# --------------------
# 4th down attempts and label
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
    st.error("No 4th down go for it attempts found in this dataset.")
    st.stop()

st.write(f"Total 4th down attempts detected: **{attempts.shape[0]:,}**")

# --------------------
# Feature engineering
# --------------------
attempts["ShortYards"] = (attempts["ToGo"] <= 2).astype(int)
attempts["InOwnHalf"] = (attempts["YardLine"] < 50).astype(int)
attempts["RedZone"] = (attempts["YardLine"] >= 80).astype(int)
attempts["LateGame"] = ((attempts["Quarter"] >= 4) & (attempts["Minute"] <= 5)).astype(int)
attempts["Clock"] = (attempts["Quarter"] * 900 - (attempts["Minute"] * 60 + attempts["Second"])).astype(int)

# Sidebar filter
st.sidebar.header("Filters")
max_togo = int(attempts["ToGo"].max())
togo_cap = st.sidebar.slider("Max yards to go", 1, max(1, max_togo), min(15, max_togo))

attempts_filt = attempts.query("ToGo <= @togo_cap").copy()
if attempts_filt.shape[0] < 50:
    st.warning("Very few plays after this filter. Consider increasing the max yards to go.")
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
# Model training
# --------------------

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
    random_state=42,
)

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

# --------------------
# Layout with tabs
# --------------------

tab_overview, tab_model, tab_charts, tab_data = st.tabs(
    ["Overview", "Model metrics", "Visuals", "Data preview"]
)

with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total plays", f"{df.shape[0]:,}")
    c2.metric("4th down attempts", f"{attempts.shape[0]:,}")
    c3.metric("Filtered attempts", f"{attempts_filt.shape[0]:,}")
    st.write("Sample of raw data:")
    st.dataframe(df.head(20))

with tab_model:
    st.subheader("Model performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("ROC AUC", f"{auc:.3f}")
    m3.metric("Brier score", f"{brier:.3f}")
    st.caption("Lower Brier is better. Around 0.21 to 0.24 is solid for this type of task.")

with tab_charts:
    st.subheader("Key visuals")

    # Pass vs Run bar
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

    fig1, ax1 = plt.subplots(figsize=(3.5, 2.5), dpi=70)

    sns.barplot(
        data=df_plot,
        x="PlayType",
        y="Rate",
        hue="PlayType",
        palette=palette_pr,
        dodge=False,
        ax=ax1,
    )
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Play type")
    ax1.set_ylabel("Conversion rate")
    ax1.set_title("4th down conversion: Pass vs Run")
    leg = ax1.get_legend()
    if leg:
        leg.remove()
    st.pyplot(fig1, use_container_width=False)

    # Prob vs ToGo scatter
    plot_df = X_test[["ToGo"]].copy()
    plot_df["prob"] = y_prob
    plot_df["Play"] = np.where(
        X_test.get("IsPass", 0) == 1, "Pass",
        np.where(X_test.get("IsRush", 0) == 1, "Run", "Other")
    )

    plot_df = plot_df.query("ToGo <= 15").copy()
    if len(plot_df) > 20000:
        plot_df = plot_df.sample(20000, random_state=42)

    fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=70)

    sns.scatterplot(
        data=plot_df,
        x="ToGo",
        y="prob",
        hue="Play",
        palette=palette_pr,
        alpha=0.35,
        s=25,
        edgecolor=None,
        ax=ax2,
    )
    ax2.set_xlim(-0.2, 15.5)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Yards to go")
    ax2.set_ylabel("Predicted conversion probability")
    ax2.set_title("Conversion probability vs yards to go")
    ax2.legend(title="", loc="upper right")
    st.pyplot(fig2, use_container_width=False)

    # Team aggressiveness vs execution
    team_rates = (
        attempts_filt.groupby("OffenseTeam")["FourthDownSuccess_simple"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "conv_rate", "count": "attempts"})
        .sort_values("conv_rate", ascending=False)
    )
    team_playstyle = team_rates.reset_index()

    fig3, ax3 = plt.subplots(figsize=(5, 3), dpi=70)

    sns.scatterplot(
        data=team_playstyle,
        x="attempts",
        y="conv_rate",
        ax=ax3,
    )
    for _, row in team_playstyle.iterrows():
        ax3.text(
            row["attempts"] + 0.3,
            row["conv_rate"],
            row["OffenseTeam"],
            fontsize=8,
        )
    ax3.set_xlabel("Total 4th down attempts")
    ax3.set_ylabel("Conversion rate")
    ax3.set_title("Team 4th down strategy: aggressiveness vs execution")
    st.pyplot(fig3, use_container_width=False)

with tab_data:
    st.subheader("4th down attempts (engineered features)")
    st.dataframe(
        attempts_filt[
            [
                "GameId", "OffenseTeam", "DefenseTeam",
                "Down", "ToGo", "YardLine",
                "IsRush", "IsPass",
                "ShortYards", "InOwnHalf",
                "RedZone", "LateGame", "Clock",
                "FourthDownSuccess_simple",
            ]
        ].head(100)
    )

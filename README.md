# nfl-4th-down-predictor

# 🏈 NFL 4th Down Conversion Predictor (2013–2024)

This project predicts the probability of converting 4th down attempts in NFL games. I started loving football recently, and one of my favorite parts of the game is when teams decide to **go for it** on 4th down. It’s bold, emotional, strategic, and sometimes completely game-changing.

So I wanted to answer a simple question:

**“Should the team go for it?”**

Using 10 years of NFL play-by-play data, I built a machine learning model that estimates the likelihood of successfully converting a 4th down attempt.

---

## 📦 Data

- **Source:** NFL play-by-play data (2013–2024)
- **Dataset Size:** ~480,000 plays
- **Filtered Down To:** Actual 4th down attempts (rush or pass, not penalties)

A valid 4th down attempt is defined as:
- `Down == 4`
- The offense either **ran or passed**
- The play is not marked `IsNoPlay`

A conversion is counted if the play resulted in a **first down or touchdown**.

---

## 🧠 Why This Project

I really like when teams go for it — it’s one of the purest expressions of football strategy.  
But even aggressive coaches sometimes misjudge when they should act.

So I set out to build a model that gives:
- A **probability of conversion** (not just yes/no)
- Based on **field situation**, not opinion or hindsight

I first tried a basic XGBoost model, but I wanted something smarter — something inspired by **Amazon’s NFL decision models**, research from analytics departments, and deeper football strategy. This led to a refined feature engineering approach that made a noticeable improvement.

---

## 🧱 Feature Engineering

The model uses situational and contextual game features:

| Feature | Meaning |
|--------|---------|
| `ToGo` | Yards needed for a first down |
| `YardLine` | Field position (0 is own end zone, 100 is opponent’s) |
| `IsRush`, `IsPass` | Type of play attempted |
| `ShortYards` | 1 if ToGo ≤ 2 yards (high conversion situations) |
| `RedZone` | 1 if YardLine ≥ 80 (inside opponent 20) |
| `InOwnHalf` | 1 if YardLine < 50 (risk zone) |
| `Clock` / `LateGame` | Pressure + urgency context |

These features reflect **real strategic decision factors**, not just raw stats.

---

## 🤖 Model

**Model:** XGBoost (Binary Classification)  
**Objective:** Predict if a 4th down play results in a conversion (0 or 1)

Key hyperparameters:
n_estimators = 600
learning_rate = 0.03
max_depth = 4
subsample = 0.8
colsample_bytree = 0.8
objective = binary:logistic

The model outputs a **probability**, which can be interpreted on a scale of:

0.0 → very unlikely to convert
1.0 → extremely likely to convert

This makes it useful for **in-game decision-making** or analytics dashboards.

---

## 📊 Results

| Metric | Score | Explanation |
|--------|------|-------------|
| **Accuracy** | **0.6227** | Predicts correct outcome ~62% of time |
| **ROC-AUC** | **0.6566** | Meaningfully separates successful vs failed attempts |
| **Brier Score** | **0.2293** | Indicates **strong probability calibration** ✅ |

These results are **quite good**, especially because football is a **high-variance sport** where small details shift outcomes.

The model not only predicts correctly — it produces **probabilities that reflect reality**, which is crucial for actual strategic use.

---

## 🧠 Insights

- NFL teams convert **about 52%** of 4th down attempts.
- **Short yardage (≤ 2 yards)** drastically increases success probability.
- Running on 4th down tends to succeed **more often than passing**.
- Field position influences risk-reward — teams are more aggressive in opponent territory.

This aligns with many modern analytics-driven coaching strategies.

---

## 🚀 Next Steps

| Idea | Purpose |
|------|---------|
| Add Expected Points Added (EPA) | Evaluate *decision quality*, not just success chance |
| Add Player/Team Strength Metrics | Personalize model for match-ups |
| Build Live Dashboard (Streamlit) | Make a real-time “Should they go for it?” tool |
| Visual Play Recommendations | Show decision confidence with heatmaps |

---

## 🌟 Final Thoughts

This project started because I just **love football**, and especially the tension and strategy of 4th down decisions.  

Digging into this data, writing feature transformations, reading research papers, and training these models helped me appreciate the sport at a **much deeper level**.

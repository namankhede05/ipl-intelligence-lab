import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATS = ["runs_left", "balls_left", "wkts_left", "target",
         "crr", "rrr", "rrr_minus_crr", "runs_last_12"]

m = pd.read_csv("data/processed/matches_clean.csv")
d = pd.read_csv("data/processed/deliveries_clean.csv")

# Super over aur DLS wale matches hatao (target alag hota hai)
so = m["super_over"].astype(str).str.upper().isin(["Y", "YES", "TRUE"])
ok = m[(~so) & (m["method"].isna())]
print("Matches used:", len(ok), "of", len(m))

d = d[d["match_id"].isin(ok["id"]) & d["inning"].isin([1, 2])].copy()
d["legal"] = (~d["extras_type"].isin(["wides", "noballs"])).astype(int)

target = d[d["inning"] == 1].groupby("match_id")["total_runs"].sum() + 1
c = d[d["inning"] == 2].copy()
c["target"] = c["match_id"].map(target)

g = c.groupby("match_id")
c["score"] = g["total_runs"].cumsum()
c["wkts"] = g["is_wicket"].cumsum()
c["balls"] = g["legal"].cumsum()
c["runs_last_12"] = g["total_runs"].transform(lambda x: x.rolling(12, min_periods=1).sum())

c["runs_left"] = c["target"] - c["score"]
c["balls_left"] = 120 - c["balls"]
c = c[(c["runs_left"] > 0) & (c["balls_left"] > 0) & (c["wkts"] < 10)].copy()

c["wkts_left"] = 10 - c["wkts"]
c["crr"] = c["score"] * 6 / c["balls"].clip(lower=1)
c["rrr"] = c["runs_left"] * 6 / c["balls_left"]
c["rrr_minus_crr"] = c["rrr"] - c["crr"]

winner = dict(zip(ok["id"], ok["winner"]))
season = dict(zip(ok["id"], ok["season"]))
c["y"] = (c["batting_team"] == c["match_id"].map(winner)).astype(int)
c["season"] = c["match_id"].map(season).astype(int)

c = c.dropna(subset=FEATS + ["y"])
train = c[c["season"] <= 2022]
test = c[c["season"] >= 2023].copy()
Xtr, ytr, Xte, yte = train[FEATS], train["y"], test[FEATS], test["y"]
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

models = {
    "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "rf": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=50,
                                 random_state=42, n_jobs=-1),
    "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42, n_jobs=-1),
}

Path("models").mkdir(exist_ok=True)
metrics = {"features": FEATS, "n_train": int(len(train)), "n_test": int(len(test)),
           "models": {}, "importance": {}, "by_phase": {}}

test["phase"] = pd.cut(test["balls_left"], bins=[0, 24, 60, 96, 120],
                       labels=["Overs 17-20", "Overs 11-16", "Overs 5-10", "Overs 1-4"])

for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    metrics["models"][name] = {
        "accuracy": float(accuracy_score(yte, p > 0.5)),
        "roc_auc": float(roc_auc_score(yte, p)),
        "log_loss": float(log_loss(yte, p)),
    }
    imp = mdl[-1].coef_[0] if name == "lr" else mdl.feature_importances_
    metrics["importance"][name] = {f: float(v) for f, v in zip(FEATS, imp)}

    ph = ((p > 0.5).astype(int) == yte.values)
    acc = pd.Series(ph, index=test.index).groupby(test["phase"], observed=True).mean()
    metrics["by_phase"][name] = {str(k): float(v) for k, v in acc.items()}

    final = clone(mdl).fit(c[FEATS], c["y"])
    joblib.dump(final, f"models/in_{name}.joblib")

json.dump(metrics, open("models/in_metrics.json", "w"), indent=2)

print("\n=== IN-MATCH RESULTS (test set: 2023-2024) ===")
print(pd.DataFrame(metrics["models"]).T.round(3))
print("\nAccuracy by match phase:")
print(pd.DataFrame(metrics["by_phase"]).round(3))
print("\nSaved: models/in_*.joblib, in_metrics.json")
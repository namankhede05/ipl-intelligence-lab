import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATS = ["elo_diff", "form_diff", "winrate_diff", "h2h_rate",
         "venue_adv_t1", "toss_won_t1", "t1_bats_first"]


def new_state():
    return {"elo": {}, "hist": {}, "wins": {}, "played": {},
            "h2h": {}, "venue": {}, "season": None}


def team_feats(s, t):
    elo = s["elo"].get(t, 1500.0)
    h = s["hist"].get(t, [])[-5:]
    form = (sum(h) + 1) / (len(h) + 2)
    rate = (s["wins"].get(t, 0) + 5) / (s["played"].get(t, 0) + 10)
    return elo, form, rate


def bat_first_team(team1, team2, toss_winner, toss_decision):
    if toss_decision == "bat":
        return toss_winner
    return team2 if toss_winner == team1 else team1


def build_row(s, team1, team2, venue, toss_winner, toss_decision):
    e1, f1, r1 = team_feats(s, team1)
    e2, f2, r2 = team_feats(s, team2)
    h = s["h2h"].get(tuple(sorted([team1, team2])), {})
    w1, w2 = h.get(team1, 0), h.get(team2, 0)
    vb, vn = s["venue"].get(venue, (0, 0))
    v_rate = (vb + 5) / (vn + 10)
    bf = bat_first_team(team1, team2, toss_winner, toss_decision)
    t1_bf = 1 if bf == team1 else 0
    return {
        "elo_diff": e1 - e2,
        "form_diff": f1 - f2,
        "winrate_diff": r1 - r2,
        "h2h_rate": (w1 + 2) / (w1 + w2 + 4),
        "venue_adv_t1": v_rate if t1_bf else 1 - v_rate,
        "toss_won_t1": int(toss_winner == team1),
        "t1_bats_first": t1_bf,
    }


def update_state(s, season, team1, team2, venue, toss_winner, toss_decision, winner):
    if s["season"] is not None and season != s["season"]:
        for t in s["elo"]:
            s["elo"][t] = 0.75 * s["elo"][t] + 0.25 * 1500
    s["season"] = season
    e1, e2 = s["elo"].get(team1, 1500.0), s["elo"].get(team2, 1500.0)
    exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
    r1 = 1.0 if winner == team1 else 0.0
    s["elo"][team1] = e1 + 20 * (r1 - exp1)
    s["elo"][team2] = e2 + 20 * ((1 - r1) - (1 - exp1))
    for t in (team1, team2):
        won = 1 if winner == t else 0
        s["hist"].setdefault(t, []).append(won)
        s["wins"][t] = s["wins"].get(t, 0) + won
        s["played"][t] = s["played"].get(t, 0) + 1
    key = tuple(sorted([team1, team2]))
    s["h2h"].setdefault(key, {})
    s["h2h"][key][winner] = s["h2h"][key].get(winner, 0) + 1
    bf = bat_first_team(team1, team2, toss_winner, toss_decision)
    vb, vn = s["venue"].get(venue, (0, 0))
    s["venue"][venue] = (vb + (1 if bf == winner else 0), vn + 1)


if __name__ == "__main__":
    m = pd.read_csv("data/processed/matches_clean.csv", parse_dates=["date"])
    m = m.sort_values(["date", "id"]).reset_index(drop=True)

    s = new_state()
    rows = []
    for r in m.itertuples():
        f = build_row(s, r.team1, r.team2, r.venue, r.toss_winner, r.toss_decision)
        f["season"] = int(r.season)
        f["y"] = int(r.winner == r.team1)
        rows.append(f)
        update_state(s, int(r.season), r.team1, r.team2, r.venue,
                     r.toss_winner, r.toss_decision, r.winner)
    df = pd.DataFrame(rows)

    train = df[df["season"] <= 2022]
    test = df[df["season"] >= 2023]
    Xtr, ytr, Xte, yte = train[FEATS], train["y"], test[FEATS], test["y"]
    print(f"Train matches: {len(train)} (2008-2022) | Test matches: {len(test)} (2023-2024)")

    base_elo = float(((test["elo_diff"] > 0).astype(int) == yte).mean())
    base_team1 = float(yte.mean())
    print(f"Baseline 1 (hamesha higher Elo wali team): {base_elo:.3f}")
    print(f"Baseline 2 (hamesha team1): {base_team1:.3f}")

    models = {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                     random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=42),
    }

    Path("models").mkdir(exist_ok=True)
    metrics = {"baseline_elo_acc": base_elo, "baseline_team1_acc": base_team1,
               "n_train": int(len(train)), "n_test": int(len(test)),
               "features": FEATS, "models": {}, "importance": {}}

    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        metrics["models"][name] = {
            "accuracy": float(accuracy_score(yte, p > 0.5)),
            "roc_auc": float(roc_auc_score(yte, p)),
            "log_loss": float(log_loss(yte, p)),
        }
        if name == "lr":
            imp = mdl[-1].coef_[0]
        else:
            imp = mdl.feature_importances_
        metrics["importance"][name] = {f: float(v) for f, v in zip(FEATS, imp)}

        final = clone(mdl).fit(df[FEATS], df["y"])
        joblib.dump(final, f"models/pre_{name}.joblib")

    joblib.dump(s, "models/pre_state.joblib")
    json.dump(metrics, open("models/pre_metrics.json", "w"), indent=2)

    print("\n=== PRE-MATCH RESULTS (test set: 2023-2024) ===")
    print(pd.DataFrame(metrics["models"]).T.round(3))
    print("\nFeature importance (xgb):")
    print(pd.Series(metrics["importance"]["xgb"]).sort_values(ascending=False).round(3))
    print("\nSaved: models/pre_*.joblib, pre_state.joblib, pre_metrics.json")
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.prematch import build_row, FEATS as PRE_FEATS

st.set_page_config(page_title="IPL Intelligence Lab", page_icon="🏏", layout="wide")

IN_FEATS = ["runs_left", "balls_left", "wkts_left", "target",
            "crr", "rrr", "rrr_minus_crr", "runs_last_12"]
EXCL = ["run out", "retired hurt", "retired out", "obstructing the field"]
MODEL_LABEL = {"avg": "Ensemble (average)", "lr": "Logistic Regression",
               "rf": "Random Forest", "xgb": "XGBoost"}
FEAT_LABEL = {"elo_diff": "Elo rating gap", "form_diff": "Recent form (last 5)",
              "winrate_diff": "Overall win rate", "h2h_rate": "Head-to-head record",
              "venue_adv_t1": "Venue bat-first/chase edge", "toss_won_t1": "Toss won by Team 1",
              "t1_bats_first": "Team 1 bats first"}

CSS = """
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.stApp {background: radial-gradient(1100px 500px at 8% -8%, rgba(34,211,165,.13), transparent),
        radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,.14), transparent), #0b1220;}
section[data-testid="stSidebar"] {background: #0d1526; border-right: 1px solid #1f2d4d;}
.hero {padding: 30px 34px; border-radius: 20px; margin-bottom: 20px;
       background: linear-gradient(120deg, #0ea5e9 0%, #22d3a5 50%, #8b5cf6 100%);}
.hero h1 {margin: 0; font-size: 2.5rem; font-weight: 800; color: #04101f;}
.hero p {margin: 6px 0 0 0; font-size: 1.05rem; color: #06182e; font-weight: 500;}
.ph {font-size: 2rem; font-weight: 800; margin-bottom: 0;
     background: linear-gradient(90deg, #22d3a5, #38bdf8); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent;}
.sub {color: #8fa3c7; margin-top: 2px; margin-bottom: 18px;}
.kpi {background: linear-gradient(160deg, #121d36, #0e182d); border: 1px solid #22335a;
      border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;}
.kl {color: #8fa3c7; font-size: .75rem; text-transform: uppercase; letter-spacing: .09em;}
.kv {font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1.25;}
.ks {color: #22d3a5; font-size: .8rem;}
.card {background: #101a30; border: 1px solid #22335a; border-radius: 14px;
       padding: 16px 20px; margin: 8px 0 14px 0;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: .85rem;}
.b-low {background: #3b1d2a; color: #fb7185;} .b-mid {background: #3a2f12; color: #fbbf24;}
.b-high {background: #12352f; color: #22d3a5;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- Data & models ----------
@st.cache_data
def load_data():
    m = pd.read_csv(ROOT / "data/processed/matches_clean.csv", parse_dates=["date"])
    d = pd.read_csv(ROOT / "data/processed/deliveries_clean.csv")
    m["season"] = m["season"].astype(int)
    m["bat_first"] = np.where(m["toss_decision"] == "bat", m["toss_winner"],
                              np.where(m["toss_winner"] == m["team1"], m["team2"], m["team1"]))
    d["legal"] = (~d["extras_type"].isin(["wides", "noballs"])).astype(int)
    d["season"] = d["match_id"].map(dict(zip(m["id"], m["season"])))
    d["bowler_wkt"] = (d["is_wicket"] == 1) & (~d["dismissal_kind"].isin(EXCL))
    return m, d


@st.cache_resource
def load_models():
    pre = {k: joblib.load(ROOT / f"models/pre_{k}.joblib") for k in ["lr", "rf", "xgb"]}
    inm = {k: joblib.load(ROOT / f"models/in_{k}.joblib") for k in ["lr", "rf", "xgb"]}
    state = joblib.load(ROOT / "models/pre_state.joblib")
    return pre, inm, state


@st.cache_data
def load_metrics():
    pm = json.load(open(ROOT / "models/pre_metrics.json"))
    im = json.load(open(ROOT / "models/in_metrics.json"))
    return pm, im


@st.cache_data
def player_lists():
    bat = D.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).index.tolist()
    bowl = D[D["bowler_wkt"]].groupby("bowler").size().sort_values(ascending=False).index.tolist()
    return bat, bowl


M, D = load_data()
PRE, INM, STATE = load_models()
PRE_M, IN_M = load_metrics()
BATTERS, BOWLERS = player_lists()

TEAMS_ALL = sorted(set(M["team1"]) | set(M["team2"]))
recent = M[M["season"] >= 2024]
ACTIVE = sorted(set(recent["team1"]) | set(recent["team2"]))
vc = M["venue"].value_counts()
VENUES = list(vc[vc >= 10].index)
so = M["super_over"].astype(str).str.upper().isin(["Y", "YES", "TRUE"])
RM = M[(~so) & (M["method"].isna())].sort_values("date", ascending=False).head(250)
REPLAY_LABELS = {int(r.id): f"{r.date:%d %b %Y} | {r.team1} vs {r.team2} ({r.season})"
                 for r in RM.itertuples()}


# ---------- Helpers ----------
def style(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def head(title, sub):
    st.markdown(f"<h2 class='ph'>{title}</h2><p class='sub'>{sub}</p>", unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(f"<div class='kpi'><div class='kl'>{l}</div><div class='kv'>{v}</div>"
                   f"<div class='ks'>{s}</div></div>", unsafe_allow_html=True)


def predict_pre(t1, t2, venue, tw, td):
    row = pd.DataFrame([build_row(STATE, t1, t2, venue, tw, td)])[PRE_FEATS]
    ps = {k: float(m.predict_proba(row)[0, 1]) for k, m in PRE.items()}
    ps["avg"] = float(np.mean([ps["lr"], ps["rf"], ps["xgb"]]))
    return ps, row


def predict_df(key, X):
    if key == "avg":
        return np.mean([INM[k].predict_proba(X)[:, 1] for k in INM], axis=0)
    return INM[key].predict_proba(X)[:, 1]


def win_prob(key, score, wkts, balls, target, last12):
    if score >= target:
        return 1.0
    if wkts >= 10 or balls >= 120:
        return 0.0
    b = max(balls, 1)
    runs_left, balls_left = target - score, 120 - balls
    crr, rrr = score * 6 / b, runs_left * 6 / balls_left
    X = pd.DataFrame([{"runs_left": runs_left, "balls_left": balls_left,
                       "wkts_left": 10 - wkts, "target": target, "crr": crr, "rrr": rrr,
                       "rrr_minus_crr": rrr - crr, "runs_last_12": last12}])[IN_FEATS]
    return float(predict_df(key, X)[0])


def replay(mid, key):
    x = D[(D["match_id"] == mid) & D["inning"].isin([1, 2])]
    tgt = int(x[x["inning"] == 1]["total_runs"].sum()) + 1
    c = x[x["inning"] == 2].copy()
    if c.empty:
        return None, tgt
    c["score"] = c["total_runs"].cumsum()
    c["wkts"] = c["is_wicket"].cumsum()
    c["balls"] = c["legal"].cumsum()
    c["last12"] = c["total_runs"].rolling(12, min_periods=1).sum()
    c = c[(c["score"] < tgt) & (c["balls"] < 120) & (c["wkts"] < 10)].copy()
    if c.empty:
        return None, tgt
    rl = tgt - c["score"]
    bl = 120 - c["balls"]
    crr = c["score"] * 6 / c["balls"].clip(lower=1)
    rrr = rl * 6 / bl
    X = pd.DataFrame({"runs_left": rl, "balls_left": bl, "wkts_left": 10 - c["wkts"],
                      "target": tgt, "crr": crr, "rrr": rrr, "rrr_minus_crr": rrr - crr,
                      "runs_last_12": c["last12"]})[IN_FEATS]
    c["prob"] = predict_df(key, X)
    c["over"] = c["balls"] / 6
    return c, tgt


# ---------- Pages ----------
def page_home():
    st.markdown("<div class='hero'><h1>🏏 IPL Intelligence Lab</h1>"
                "<p>From raw data to winning insights: analytics, machine learning aur live "
                "win probability, sab ek jagah.</p></div>", unsafe_allow_html=True)
    kpi_row([("Matches", f"{len(M):,}", "2008 - 2024"),
             ("Ball-by-ball rows", f"{len(D):,}", "deliveries"),
             ("Teams", len(TEAMS_ALL), "all-time"),
             ("Players (batters)", f"{D['batter'].nunique():,}", "in dataset")])

    t = D[D["inning"] == 1].groupby("match_id")["total_runs"].sum().rename("runs").reset_index()
    t = t.merge(M[["id", "season"]], left_on="match_id", right_on="id")
    avg = t.groupby("season")["runs"].mean().round(1).reset_index()
    c1, c2 = st.columns(2)
    f1 = px.line(avg, x="season", y="runs", markers=True, color_discrete_sequence=["#f59e0b"],
                 title="Average 1st innings score per season")
    c1.plotly_chart(style(f1, 360), use_container_width=True)
    titles = M[M["match_type"] == "Final"]["winner"].value_counts().reset_index()
    titles.columns = ["team", "titles"]
    if len(titles):
        f2 = px.bar(titles.iloc[::-1], x="titles", y="team", orientation="h", text="titles",
                    color_discrete_sequence=["#22d3a5"], title="IPL titles (Final winners)")
        c2.plotly_chart(style(f2, 360), use_container_width=True)

    field = M[M["toss_decision"] == "field"]
    fp = 100 * (field["toss_winner"] == field["winner"]).mean()
    st.markdown(f"""<div class='card'><b>Key insights</b><br>
    • Average 1st innings score {avg['runs'].iloc[0]} (2008) se {avg['runs'].iloc[-1]} ({int(avg['season'].iloc[-1])}) tak badh gaya.<br>
    • Toss jeet kar <b>field</b> choose karne wali team {fp:.1f}% matches jeeti.<br>
    • In-match model chase ke 17-20 overs mein ~89% accuracy deta hai; pre-match prediction lagbhag coin-flip hai.<br>
    • Left sidebar se pages badlo: Predictor, Live Win Probability, What-if Simulator aur Model Performance.</div>""",
                unsafe_allow_html=True)


def page_team():
    head("📊 Team Analytics", "Team ka poora IPL record")
    team = st.selectbox("Team chuno", TEAMS_ALL, index=TEAMS_ALL.index("Chennai Super Kings"))
    mt = M[(M["team1"] == team) | (M["team2"] == team)]

    def wp(df):
        return f"{100 * (df['winner'] == team).mean():.1f}%" if len(df) else "-"

    titles = int(((M["match_type"] == "Final") & (M["winner"] == team)).sum())
    kpi_row([("Matches", len(mt), ""), ("Wins", int((mt["winner"] == team).sum()), wp(mt) + " win"),
             ("Titles", titles, "Final wins"),
             ("Bat first win %", wp(mt[mt["bat_first"] == team]), ""),
             ("Chase win %", wp(mt[mt["bat_first"] != team]), "")])

    rows = [{"season": s, "win_pct": round(100 * (g["winner"] == team).mean(), 1)}
            for s, g in mt.groupby("season")]
    f = px.bar(pd.DataFrame(rows), x="season", y="win_pct", text="win_pct",
               color="win_pct", color_continuous_scale="Viridis", title="Season-wise win %")
    st.plotly_chart(style(f, 380), use_container_width=True)

    c1, c2 = st.columns(2)
    bat = D[D["batting_team"] == team].groupby("batter")["batsman_runs"].sum().nlargest(10)
    f1 = px.bar(bat.iloc[::-1], orientation="h", text_auto=True,
                color_discrete_sequence=["#38bdf8"], title="Top run scorers for team")
    f1.update_layout(showlegend=False, xaxis_title="runs", yaxis_title="")
    c1.plotly_chart(style(f1, 400), use_container_width=True)
    bw = D[(D["bowling_team"] == team) & D["bowler_wkt"]].groupby("bowler").size().nlargest(10)
    f2 = px.bar(bw.iloc[::-1], orientation="h", text_auto=True,
                color_discrete_sequence=["#fb7185"], title="Top wicket takers for team")
    f2.update_layout(showlegend=False, xaxis_title="wickets", yaxis_title="")
    c2.plotly_chart(style(f2, 400), use_container_width=True)


def page_player():
    head("🧑‍🎓 Player Stats", "Batter aur bowler ka career record")
    t1, t2 = st.tabs(["🏏 Batter", "🎯 Bowler"])
    with t1:
        name = st.selectbox("Batter chuno", BATTERS, key="bat_sel")
        b = D[D["batter"] == name]
        runs = int(b["batsman_runs"].sum())
        balls = int((b["extras_type"] != "wides").sum())
        pm = b.groupby("match_id")["batsman_runs"].sum()
        outs = int((D["player_dismissed"] == name).sum())
        kpi_row([("Runs", f"{runs:,}", f"{len(pm)} innings"),
                 ("Strike rate", f"{100 * runs / max(balls, 1):.1f}", f"{balls} balls"),
                 ("Average", f"{runs / max(outs, 1):.1f}", f"{outs} dismissals"),
                 ("Highest", int(pm.max()), ""),
                 ("50s / 100s", f"{int(((pm >= 50) & (pm < 100)).sum())} / {int((pm >= 100).sum())}", ""),
                 ("4s / 6s", f"{int((b['batsman_runs'] == 4).sum())} / {int((b['batsman_runs'] == 6).sum())}", "")])
        s = b.groupby("season")["batsman_runs"].sum().reset_index()
        f = px.bar(s, x="season", y="batsman_runs", text="batsman_runs",
                   color_discrete_sequence=["#38bdf8"], title=f"{name}: season-wise runs")
        st.plotly_chart(style(f, 380), use_container_width=True)
    with t2:
        name = st.selectbox("Bowler chuno", BOWLERS, key="bowl_sel")
        w = D[D["bowler"] == name]
        legal = int(w["legal"].sum())
        conceded = int(w["batsman_runs"].sum() +
                       w.loc[w["extras_type"].isin(["wides", "noballs"]), "extra_runs"].sum())
        wk = w[w["bowler_wkt"]]
        n = len(wk)
        kpi_row([("Wickets", n, f"{w['match_id'].nunique()} matches"),
                 ("Economy", f"{conceded * 6 / max(legal, 1):.2f}", f"{legal} balls"),
                 ("Average", f"{conceded / max(n, 1):.1f}", "runs / wicket"),
                 ("Strike rate", f"{legal / max(n, 1):.1f}", "balls / wicket")])
        s = wk.groupby("season").size().reset_index(name="wickets")
        f = px.bar(s, x="season", y="wickets", text="wickets",
                   color_discrete_sequence=["#fb7185"], title=f"{name}: season-wise wickets")
        st.plotly_chart(style(f, 380), use_container_width=True)


def page_h2h():
    head("⚔️ Head-to-Head", "Do teams ka aamne-saamne record")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Team A", TEAMS_ALL, index=TEAMS_ALL.index("Chennai Super Kings"))
    b = c2.selectbox("Team B", TEAMS_ALL, index=TEAMS_ALL.index("Mumbai Indians"))
    if a == b:
        st.warning("Do alag teams chuno.")
        return
    h = M[((M["team1"] == a) & (M["team2"] == b)) | ((M["team1"] == b) & (M["team2"] == a))]
    wa, wb = int((h["winner"] == a).sum()), int((h["winner"] == b).sum())
    kpi_row([("Matches", len(h), ""), (a, wa, "wins"), (b, wb, "wins")])
    if len(h) == 0:
        st.info("In dono teams ke beech koi match nahi mila.")
        return
    c1, c2 = st.columns([1, 1.4])
    f = px.pie(names=[a, b], values=[wa, wb], hole=.6,
               color_discrete_sequence=["#22d3a5", "#8b5cf6"], title="Win share")
    c1.plotly_chart(style(f, 360), use_container_width=True)
    c2.markdown("**Season-wise jeet**")
    c2.dataframe(pd.crosstab(h["season"], h["winner"]), use_container_width=True, height=300)
    st.markdown("**Last 8 matches**")
    last = h.sort_values("date", ascending=False).head(8)[
        ["date", "venue", "winner", "result_margin", "result"]].copy()
    last["date"] = last["date"].dt.date
    st.dataframe(last, use_container_width=True, hide_index=True)


def page_predictor():
    head("🎯 Match Predictor", "Toss ke baad, match shuru hone se pehle ka prediction")
    c1, c2, c3 = st.columns(3)
    t1 = c1.selectbox("Team 1", ACTIVE, index=0)
    t2 = c2.selectbox("Team 2", [t for t in ACTIVE if t != t1], index=0)
    venue = c3.selectbox("Venue", VENUES)
    c4, c5 = st.columns(2)
    tw = c4.selectbox("Toss kisne jeeta", [t1, t2])
    td = c5.selectbox("Toss ke baad decision", ["field", "bat"])

    ps, row = predict_pre(t1, t2, venue, tw, td)
    p = ps["avg"]
    conf = abs(p - 0.5) * 2
    lab, cls = (("LOW", "b-low") if conf < 0.2 else ("MEDIUM", "b-mid") if conf < 0.5
                else ("HIGH", "b-high"))
    kpi_row([(t1, f"{p * 100:.1f}%", "win probability"), (t2, f"{(1 - p) * 100:.1f}%", "win probability"),
             ("Elo " + t1.split()[0], f"{STATE['elo'].get(t1, 1500):.0f}", ""),
             ("Elo " + t2.split()[0], f"{STATE['elo'].get(t2, 1500):.0f}", "")])
    st.progress(p)
    acc = PRE_M["models"]["xgb"]["accuracy"] * 100
    st.markdown(f"<div class='card'>Prediction confidence: <span class='badge {cls}'>{lab}</span><br>"
                f"<b>Limitation:</b> Pre-match model ki test accuracy sirf ~{acc:.0f}% rahi "
                f"(2023-24 held-out). Matlab ye coin-flip ke kareeb hai, kyunki cricket mein toss ke "
                f"baad bhi result bahut random hota hai. Isko betting ya final decision ke liye use "
                f"mat karo, ye ek educational estimate hai.</div>", unsafe_allow_html=True)

    pipe = PRE["lr"]
    contrib = pipe[-1].coef_[0] * pipe[0].transform(row)[0]
    df = pd.DataFrame({"factor": [FEAT_LABEL[f] for f in PRE_FEATS], "impact": contrib})
    df["favours"] = np.where(df["impact"] >= 0, t1, t2)
    df = df.reindex(df["impact"].abs().sort_values().index)
    f = px.bar(df, x="impact", y="factor", orientation="h", color="favours",
               color_discrete_sequence=["#22d3a5", "#8b5cf6"],
               title="Top factors (Logistic Regression explanation)")
    st.plotly_chart(style(f, 380), use_container_width=True)
    st.markdown("**Model-wise probability (Team 1 jeetegi)**")
    st.dataframe(pd.DataFrame({"model": [MODEL_LABEL[k] for k in ["lr", "rf", "xgb", "avg"]],
                               "team1_win_prob": [f"{ps[k] * 100:.1f}%" for k in ["lr", "rf", "xgb", "avg"]]}),
                 hide_index=True, use_container_width=True)


def page_live():
    head("📡 Live Win Probability", "Chase ke dauran har ball par jeetne ka chance")
    tab1, tab2 = st.tabs(["🧮 Live Calculator", "🎬 Match Replay"])
    with tab1:
        a, b, c = st.columns(3)
        target = a.number_input("Target (runs)", 60, 300, 180)
        score = b.number_input("Current score", 0, 300, 95)
        wkts = c.number_input("Wickets gire", 0, 9, 3)
        d1, d2, d3 = st.columns(3)
        ov = d1.number_input("Overs complete", 0, 19, 11)
        bl = d2.number_input("Balls in current over", 0, 5, 2)
        l12 = d3.number_input("Last 12 balls mein runs", 0, 72, 16)
        mk = st.selectbox("Model", list(MODEL_LABEL), format_func=lambda k: MODEL_LABEL[k],
                          key="live_model")
        balls = int(ov * 6 + bl)
        p = win_prob(mk, score, wkts, balls, target, l12)
        left = target - score
        kpi_row([("Runs needed", max(left, 0), ""), ("Balls left", 120 - balls, ""),
                 ("Current RR", f"{score * 6 / max(balls, 1):.2f}", ""),
                 ("Required RR", f"{max(left, 0) * 6 / max(120 - balls, 1):.2f}", "")])
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Chasing team win probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22d3a5"},
                   "steps": [{"range": [0, 50], "color": "#3b1d2a"},
                             {"range": [50, 100], "color": "#12352f"}]}))
        st.plotly_chart(style(g, 340), use_container_width=True)
    with tab2:
        st.caption("Note: final models poore data par trained hain, isliye replay illustration hai. "
                   "Honest accuracy ke liye Model Performance page dekho.")
        c1, c2 = st.columns([2, 1])
        mid = c1.selectbox("Match chuno", list(REPLAY_LABELS), format_func=lambda i: REPLAY_LABELS[i])
        key = c2.selectbox("Model", list(MODEL_LABEL), index=0,
                           format_func=lambda k: MODEL_LABEL[k], key="replay_model")
        c, tgt = replay(mid, key)
        win = M.loc[M["id"] == mid, "winner"].iloc[0]
        if c is None:
            st.info("Is match ka replay available nahi hai.")
        else:
            chasing = c["batting_team"].iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=c["over"], y=c["prob"] * 100, mode="lines",
                                     line=dict(color="#22d3a5", width=3), name="Win %",
                                     fill="tozeroy", fillcolor="rgba(34,211,165,.12)"))
            w = c[c["is_wicket"] == 1]
            fig.add_trace(go.Scatter(x=w["over"], y=w["prob"] * 100, mode="markers",
                                     marker=dict(color="#fb7185", size=10), name="Wicket"))
            fig.add_hline(y=50, line_dash="dot", line_color="#64748b")
            fig.update_layout(title=f"{chasing} chasing {tgt}: win probability",
                              xaxis_title="Overs", yaxis_title="Win %", yaxis_range=[0, 100])
            st.plotly_chart(style(fig, 440), use_container_width=True)
            st.success(f"Match winner: {win}")


def page_whatif():
    head("🧪 What-if Simulator", "Halaat badlo, prediction kaise badalti hai dekho")
    tab1, tab2 = st.tabs(["🏃 Chase what-if", "🏟️ Pre-match what-if"])
    with tab1:
        a, b, c = st.columns(3)
        target = a.number_input("Target", 60, 300, 180, key="w_t")
        score = b.number_input("Score", 0, 300, 100, key="w_s")
        wk = c.number_input("Wickets", 0, 9, 3, key="w_w")
        d1, d2, d3 = st.columns(3)
        ov = d1.number_input("Overs complete", 0, 18, 12, key="w_o")
        bl = d2.number_input("Balls in over", 0, 5, 0, key="w_b")
        l12 = d3.number_input("Last 12 balls runs", 0, 72, 16, key="w_l")
        mk = st.selectbox("Model", list(MODEL_LABEL), format_func=lambda k: MODEL_LABEL[k], key="w_m")
        balls = int(ov * 6 + bl)
        base = win_prob(mk, score, wk, balls, target, l12)
        scen = [("Dot-ball over (2 runs)", 2, 0), ("Normal over (8 runs)", 8, 0),
                ("Big over (16 runs)", 16, 0), ("Wicket + 6 runs", 6, 1),
                ("2 wickets + 4 runs", 4, 2), ("Collapse: 3 wkts, 2 runs", 2, 3)]
        rows = []
        for name, r, w in scen:
            p = win_prob(mk, score + r, wk + w, balls + 6, target, round(l12 / 2 + r))
            rows.append({"Next over": name, "Win %": round(p * 100, 1),
                         "Change": round((p - base) * 100, 1)})
        df = pd.DataFrame(rows)
        st.markdown(f"**Abhi ka win probability: {base * 100:.1f}%**")
        f = px.bar(df, x="Next over", y="Win %", text="Win %", color="Change",
                   color_continuous_scale="RdYlGn", title="Agla over aise gaya to...")
        f.add_hline(y=base * 100, line_dash="dot", line_color="#fff")
        st.plotly_chart(style(f, 400), use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
    with tab2:
        c1, c2, c3 = st.columns(3)
        t1 = c1.selectbox("Team 1", ACTIVE, index=0, key="p_t1")
        t2 = c2.selectbox("Team 2", [t for t in ACTIVE if t != t1], index=0, key="p_t2")
        venue = c3.selectbox("Venue", VENUES, key="p_v")
        rows = []
        for who in (t1, t2):
            for dec in ("bat", "field"):
                p = predict_pre(t1, t2, venue, who, dec)[0]["avg"]
                rows.append({"Scenario": f"{who} wins toss, {dec}s", "Team 1 win %": round(p * 100, 1)})
        f = px.bar(pd.DataFrame(rows), x="Scenario", y="Team 1 win %", text="Team 1 win %",
                   color_discrete_sequence=["#8b5cf6"], title=f"{t1} ki jeet ka chance: toss scenarios")
        f.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(style(f, 380), use_container_width=True)
        vr = [{"Venue": v, "Team 1 win %": round(predict_pre(t1, t2, v, t1, "field")[0]["avg"] * 100, 1)}
              for v in VENUES[:10]]
        f2 = px.bar(pd.DataFrame(vr), x="Team 1 win %", y="Venue", orientation="h",
                    text="Team 1 win %", color_discrete_sequence=["#22d3a5"],
                    title="Alag venues par (Team 1 toss jeet kar field karti hai)")
        f2.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(style(f2, 420), use_container_width=True)
        st.caption("Pre-match models kamzor hain, isliye badlav chhote dikhte hain. "
                   "Ye batata hai ki model context ko kaise dekhta hai, asli result nahi.")


def page_perf():
    head("📈 Model Performance", "Honest evaluation: train 2008-2022, test 2023-2024 (no data leakage)")
    t1, t2 = st.tabs(["In-match (chase)", "Pre-match"])
    with t1:
        df = pd.DataFrame(IN_M["models"]).T
        df.index = [MODEL_LABEL[i] for i in df.index]
        kpi_row([("Train rows", f"{IN_M['n_train']:,}", "2008-2022"),
                 ("Test rows", f"{IN_M['n_test']:,}", "2023-2024")])
        c1, c2 = st.columns(2)
        f = px.bar(df[["accuracy", "roc_auc"]], barmode="group", text_auto=".3f",
                   title="Accuracy & ROC-AUC", color_discrete_sequence=["#22d3a5", "#8b5cf6"])
        c1.plotly_chart(style(f, 380), use_container_width=True)
        ph = pd.DataFrame(IN_M["by_phase"]).loc[["Overs 1-4", "Overs 5-10", "Overs 11-16", "Overs 17-20"]]
        ph.columns = [MODEL_LABEL[c] for c in ph.columns]
        f2 = px.line(ph, markers=True, title="Accuracy by match phase")
        c2.plotly_chart(style(f2, 380), use_container_width=True)
        imp = pd.Series(IN_M["importance"]["xgb"]).sort_values()
        f3 = px.bar(imp, orientation="h", color_discrete_sequence=["#f59e0b"],
                    title="Feature importance (XGBoost)")
        f3.update_layout(showlegend=False, xaxis_title="importance", yaxis_title="")
        st.plotly_chart(style(f3, 380), use_container_width=True)
        st.dataframe(df.round(3), use_container_width=True)
        best = df["roc_auc"].idxmax()
        st.markdown(f"<div class='card'><b>Resume bullet (real numbers):</b><br>"
                    f"Built in-match win-probability model on 260K+ IPL ball-by-ball records; "
                    f"{best} achieved {df.loc[best, 'accuracy'] * 100:.1f}% accuracy and "
                    f"{df.loc[best, 'roc_auc']:.2f} ROC-AUC on time-split held-out seasons "
                    f"(2023-24), with no data leakage.</div>", unsafe_allow_html=True)
    with t2:
        df = pd.DataFrame(PRE_M["models"]).T
        df.index = [MODEL_LABEL[i] for i in df.index]
        kpi_row([("Baseline: higher Elo", f"{PRE_M['baseline_elo_acc'] * 100:.1f}%", "accuracy"),
                 ("Baseline: always Team 1", f"{PRE_M['baseline_team1_acc'] * 100:.1f}%", "accuracy"),
                 ("Test matches", PRE_M["n_test"], "2023-2024")])
        st.dataframe(df.round(3), use_container_width=True)
        imp = pd.Series(PRE_M["importance"]["xgb"]).rename(index=FEAT_LABEL).sort_values()
        f = px.bar(imp, orientation="h", color_discrete_sequence=["#8b5cf6"],
                   title="Feature importance (XGBoost)")
        f.update_layout(showlegend=False, xaxis_title="importance", yaxis_title="")
        st.plotly_chart(style(f, 360), use_container_width=True)
        st.markdown("<div class='card'><b>Honest takeaway:</b> Pre-match models baseline se behtar "
                    "nahi nikle (ROC-AUC ~0.52). Matlab sirf team strength aur toss se IPL ka result "
                    "predict karna lagbhag mushkil hai. Player-level (playing XI) data jaisi features "
                    "future scope hain.</div>", unsafe_allow_html=True)


# ---------- Navigation ----------
PAGES = {"🏠 Home": page_home, "📊 Team Analytics": page_team, "🧑‍🎓 Player Stats": page_player,
         "⚔️ Head-to-Head": page_h2h, "🎯 Match Predictor": page_predictor,
         "📡 Live Win Probability": page_live, "🧪 What-if Simulator": page_whatif,
         "📈 Model Performance": page_perf}

st.sidebar.markdown("## 🏏 IPL Intelligence Lab")
st.sidebar.caption("Data + Analytics + ML + Dashboard")
choice = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Educational project. Predictions estimates hain, guaranteed outcomes nahi.")
PAGES[choice]()
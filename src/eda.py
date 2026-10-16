import sqlite3
import webbrowser
import pandas as pd
import plotly.express as px
from pathlib import Path

con = sqlite3.connect("data/ipl.db")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

def q(sql):
    return pd.read_sql_query(sql, con)

figs = []
def add(fig, title):
    fig.update_layout(template="plotly_dark", title=title, height=480,
                      paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      margin=dict(l=40, r=20, t=70, b=40))
    figs.append(fig)

# 1. Season-wise matches
d = q("SELECT season, COUNT(*) AS matches FROM matches GROUP BY season ORDER BY season")
add(px.bar(d, x="season", y="matches", text="matches", color_discrete_sequence=["#22d3a5"]),
    "Season-wise matches")

# 2. Average first innings score
d = q("""SELECT m.season, ROUND(AVG(t.total),1) AS avg_score FROM
         (SELECT match_id, SUM(total_runs) AS total FROM deliveries WHERE inning=1 GROUP BY match_id) t
         JOIN matches m ON t.match_id = m.id GROUP BY m.season ORDER BY m.season""")
add(px.line(d, x="season", y="avg_score", markers=True, color_discrete_sequence=["#f59e0b"]),
    "Average first innings score per season (scoring badh raha hai)")
best = d.loc[d["avg_score"].idxmax()]
print("Sabse high avg score season:", int(best["season"]), best["avg_score"])

# 3. Team win %
d = q("""WITH p AS (SELECT team1 AS team, winner FROM matches UNION ALL SELECT team2, winner FROM matches)
         SELECT team, COUNT(*) AS played,
         ROUND(100.0*SUM(CASE WHEN team=winner THEN 1 ELSE 0 END)/COUNT(*),1) AS win_pct
         FROM p GROUP BY team HAVING COUNT(*)>=30 ORDER BY win_pct""")
add(px.bar(d, x="win_pct", y="team", orientation="h", text="win_pct", color="win_pct",
           color_continuous_scale="Viridis"), "Team-wise win % (min 30 matches)")

# 4. Toss decision aur result
d = q("""SELECT toss_decision, COUNT(*) AS matches,
         ROUND(100.0*SUM(CASE WHEN toss_winner=winner THEN 1 ELSE 0 END)/COUNT(*),1) AS toss_winner_won_pct
         FROM matches GROUP BY toss_decision""")
add(px.bar(d, x="toss_decision", y="toss_winner_won_pct", text="toss_winner_won_pct",
           color_discrete_sequence=["#8b5cf6"]), "Toss jeetne ke baad bat/field choose karne par win %")

# 5. Top 10 run scorers
d = q("SELECT batter, SUM(batsman_runs) AS runs FROM deliveries GROUP BY batter ORDER BY runs DESC LIMIT 10")
add(px.bar(d.iloc[::-1], x="runs", y="batter", orientation="h", text="runs",
           color_discrete_sequence=["#38bdf8"]), "Top 10 run scorers")

# 6. Top 10 wicket takers
d = q("""SELECT bowler, COUNT(*) AS wickets FROM deliveries WHERE is_wicket=1
         AND dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
         GROUP BY bowler ORDER BY wickets DESC LIMIT 10""")
add(px.bar(d.iloc[::-1], x="wickets", y="bowler", orientation="h", text="wickets",
           color_discrete_sequence=["#fb7185"]), "Top 10 wicket takers")

# 7. Venue: bat first win %
d = q("""WITH x AS (SELECT venue,
           CASE WHEN toss_decision='bat' THEN toss_winner
                WHEN toss_winner=team1 THEN team2 ELSE team1 END AS bat_first, winner FROM matches)
         SELECT venue, COUNT(*) AS matches,
         ROUND(100.0*SUM(CASE WHEN bat_first=winner THEN 1 ELSE 0 END)/COUNT(*),1) AS bat_first_win_pct
         FROM x GROUP BY venue HAVING COUNT(*)>=20 ORDER BY bat_first_win_pct""")
add(px.bar(d, x="bat_first_win_pct", y="venue", orientation="h", text="bat_first_win_pct",
           color="bat_first_win_pct", color_continuous_scale="RdYlGn"),
    "Venue-wise: pehle batting karne wali team ka win % (min 20 matches)")

# 8. Head-to-head heatmap
main = ["Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders", "Rajasthan Royals",
        "Royal Challengers Bengaluru", "Sunrisers Hyderabad", "Delhi Capitals", "Punjab Kings"]
m = q("SELECT team1, team2, winner FROM matches")
mat = pd.DataFrame(0, index=main, columns=main)
for _, r in m.iterrows():
    if r.team1 in main and r.team2 in main and r.winner in (r.team1, r.team2):
        loser = r.team2 if r.winner == r.team1 else r.team1
        mat.loc[r.winner, loser] += 1
fig = px.imshow(mat, text_auto=True, color_continuous_scale="Viridis",
                labels=dict(x="Opponent (haara)", y="Team (jeeta)", color="Wins"))
fig.update_layout(height=620)
add(fig, "Head-to-head: row wali team ne column wali team ko kitni baar haraya")

# 9. Runs per over
d = q("SELECT over_no, ROUND(6.0*SUM(total_runs)/COUNT(*),2) AS runs_per_over FROM deliveries GROUP BY over_no ORDER BY over_no")
if d["over_no"].min() == 0:
    d["over_no"] = d["over_no"] + 1
add(px.line(d, x="over_no", y="runs_per_over", markers=True, color_discrete_sequence=["#22d3a5"]),
    "Over-wise scoring rate (death overs mein run rate sabse zyada)")

# 10. Player of the match
d = q("""SELECT player_of_match AS player, COUNT(*) AS awards FROM matches
         WHERE player_of_match IS NOT NULL GROUP BY player_of_match ORDER BY awards DESC LIMIT 10""")
add(px.bar(d.iloc[::-1], x="awards", y="player", orientation="h", text="awards",
           color_discrete_sequence=["#f59e0b"]), "Top 10 Player of the Match winners")

# Report banao
html = ("<html><head><meta charset='utf-8'><title>IPL EDA</title></head>"
        "<body style='background:#0b1220;color:#fff;font-family:Segoe UI,sans-serif;padding:20px'>"
        "<h1>IPL Intelligence Lab - EDA Report</h1>")
for i, f in enumerate(figs):
    html += f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
html += "</body></html>"

path = OUT / "eda_report.html"
path.write_text(html, encoding="utf-8")
print("Report ready:", path)
webbrowser.open(path.resolve().as_uri())
con.close()
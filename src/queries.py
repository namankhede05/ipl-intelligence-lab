import sqlite3
import pandas as pd

con = sqlite3.connect("data/ipl.db")
pd.set_option("display.width", 200)

QUERIES = {

"1. Season-wise matches aur sabse zyada jeetne wali team": """
SELECT season, COUNT(*) AS matches,
  (SELECT winner FROM matches m2 WHERE m2.season = m.season
   GROUP BY winner ORDER BY COUNT(*) DESC LIMIT 1) AS most_wins_team
FROM matches m
GROUP BY season
ORDER BY season;
""",

"2. Team-wise win % (minimum 50 matches)": """
WITH played AS (
  SELECT team1 AS team, winner FROM matches
  UNION ALL
  SELECT team2 AS team, winner FROM matches
)
SELECT team, COUNT(*) AS played,
  SUM(CASE WHEN team = winner THEN 1 ELSE 0 END) AS wins,
  ROUND(100.0 * SUM(CASE WHEN team = winner THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct
FROM played
GROUP BY team
HAVING COUNT(*) >= 50
ORDER BY win_pct DESC;
""",

"3. Toss jeetne se match jeetne ka chance": """
SELECT toss_decision, COUNT(*) AS matches,
  ROUND(100.0 * SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) / COUNT(*), 1) AS toss_winner_won_pct
FROM matches
GROUP BY toss_decision;
""",

"4. Top 10 run scorers (strike rate ke saath)": """
SELECT batter, runs, balls, ROUND(100.0 * runs / balls, 1) AS strike_rate
FROM (
  SELECT batter, SUM(batsman_runs) AS runs,
    SUM(CASE WHEN extras_type = 'wides' THEN 0 ELSE 1 END) AS balls
  FROM deliveries
  GROUP BY batter
)
ORDER BY runs DESC
LIMIT 10;
""",

"5. Top 10 wicket takers": """
SELECT bowler, COUNT(*) AS wickets
FROM deliveries
WHERE is_wicket = 1
  AND dismissal_kind NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
GROUP BY bowler
ORDER BY wickets DESC
LIMIT 10;
""",

"6. Venue analysis: pehle batting karne wali team ka win %": """
WITH x AS (
  SELECT venue,
    CASE WHEN toss_decision = 'bat' THEN toss_winner
         WHEN toss_winner = team1 THEN team2 ELSE team1 END AS bat_first,
    winner
  FROM matches
)
SELECT venue, COUNT(*) AS matches,
  ROUND(100.0 * SUM(CASE WHEN bat_first = winner THEN 1 ELSE 0 END) / COUNT(*), 1) AS bat_first_win_pct
FROM x
GROUP BY venue
HAVING COUNT(*) >= 20
ORDER BY matches DESC;
""",

"7. Head-to-head: CSK vs MI": """
SELECT winner, COUNT(*) AS wins
FROM matches
WHERE (team1 = 'Chennai Super Kings' AND team2 = 'Mumbai Indians')
   OR (team1 = 'Mumbai Indians' AND team2 = 'Chennai Super Kings')
GROUP BY winner
ORDER BY wins DESC;
""",

"8. Har season ka top run scorer (Window function RANK)": """
WITH s AS (
  SELECT m.season, d.batter, SUM(d.batsman_runs) AS runs
  FROM deliveries d
  JOIN matches m ON d.match_id = m.id
  GROUP BY m.season, d.batter
),
r AS (
  SELECT *, RANK() OVER (PARTITION BY season ORDER BY runs DESC) AS rnk
  FROM s
)
SELECT season, batter, runs
FROM r
WHERE rnk = 1
ORDER BY season;
""",

"9. Har season ka average first innings score": """
WITH inn AS (
  SELECT match_id, SUM(total_runs) AS total
  FROM deliveries
  WHERE inning = 1
  GROUP BY match_id
)
SELECT m.season, ROUND(AVG(inn.total), 1) AS avg_first_innings_score
FROM inn
JOIN matches m ON inn.match_id = m.id
GROUP BY m.season
ORDER BY m.season;
""",
}

for title, sql in QUERIES.items():
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    try:
        print(pd.read_sql_query(sql, con).to_string(index=False))
    except Exception as e:
        print("ERROR:", e)

con.close()
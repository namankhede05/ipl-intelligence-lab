import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(RAW / "matches.csv")
deliveries = pd.read_csv(RAW / "deliveries.csv")

print("MATCHES columns:", matches.columns.tolist())
print("DELIVERIES columns:", deliveries.columns.tolist())
print("Shapes:", matches.shape, deliveries.shape)

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

for col in ["team1", "team2", "toss_winner", "winner"]:
    if col in matches.columns:
        matches[col] = matches[col].replace(TEAM_MAP)

for col in ["batting_team", "bowling_team"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].replace(TEAM_MAP)

matches["date"] = pd.to_datetime(matches["date"])
matches = matches.drop_duplicates(subset="id").sort_values("date").reset_index(drop=True)

print("\nMissing values (matches):")
print(matches.isna().sum()[matches.isna().sum() > 0])

matches_valid = matches[matches["winner"].notna()].copy()
print("\nTotal matches:", len(matches), "| With result:", len(matches_valid))

deliveries = deliveries[deliveries["match_id"].isin(matches_valid["id"])]

matches_valid.to_csv(OUT / "matches_clean.csv", index=False)
deliveries.to_csv(OUT / "deliveries_clean.csv", index=False)
print("\nSaved clean files in data/processed/")
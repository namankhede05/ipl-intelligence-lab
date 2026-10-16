import pandas as pd
from pathlib import Path

p = Path("data/processed/matches_clean.csv")
m = pd.read_csv(p)

SEASON_MAP = {"2007/08": 2008, "2009/10": 2010, "2020/21": 2020}
m["season"] = m["season"].astype(str).map(lambda s: int(SEASON_MAP.get(s, s)))

VENUE_MAP = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association Stadium",
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",
}
m["venue"] = m["venue"].astype(str).map(lambda v: v.split(",")[0].strip())
m["venue"] = m["venue"].replace(VENUE_MAP)

m.to_csv(p, index=False)
print("Seasons:", sorted(m["season"].unique()))
print("\nTop venues:")
print(m["venue"].value_counts().head(15))
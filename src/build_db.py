import sqlite3
import pandas as pd
from pathlib import Path

P = Path("data/processed")
DB = Path("data/ipl.db")

matches = pd.read_csv(P / "matches_clean.csv")
deliveries = pd.read_csv(P / "deliveries_clean.csv")

# 'over' SQL ka keyword hai, isliye naam badal rahe hain
deliveries = deliveries.rename(columns={"over": "over_no", "ball": "ball_no"})

if DB.exists():
    DB.unlink()

con = sqlite3.connect(DB)
matches.to_sql("matches", con, index=False)
deliveries.to_sql("deliveries", con, index=False)
con.execute("CREATE INDEX idx_d_match ON deliveries(match_id)")
con.execute("CREATE INDEX idx_m_id ON matches(id)")
con.commit()

for t in ["matches", "deliveries"]:
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
    print(f"\n{t.upper()}: {n} rows")
    print("Columns:", cols)

con.close()
print("\nDatabase ready: data/ipl.db")
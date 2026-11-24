# 🏏 IPL Intelligence Lab

IPL data analytics + win probability dashboard (2008-2024).

**Live app:** (deploy ke baad link yahan daalna)

## Tech
Python, Pandas, SQLite, SQL (CTE, window functions), Plotly, scikit-learn, XGBoost, Streamlit

## What it does
- Cleaned 1,090 matches and 260K+ ball-by-ball records, standardised team and venue names
- SQL analysis (JOINs, CTEs, RANK window functions) on SQLite
- 10 EDA charts and an 8-page interactive dashboard
- **In-match win probability** (2nd innings): ~78% accuracy, 0.87 ROC-AUC on held-out 2023-24 seasons (about 89% in overs 17-20)
- **Pre-match prediction:** ROC-AUC about 0.52, so close to a coin flip. Shown honestly as low confidence in the app.
- Time-aware split (train 2008-2022, test 2023-2024), no data leakage

## Run locally
    pip install -r requirements.txt
    streamlit run app/app.py

## Project structure
- `src/` cleaning, SQL, EDA, model training
- `models/` trained models and metrics
- `app/app.py` Streamlit dashboard

*Educational project. Predictions are estimates, not guaranteed outcomes.*

**IPL Intelligence Lab – Interactive Cricket Analytics Dashboard**

🌐 **Live Website:** [Open IPL Intelligence Lab](https://ipl-intelligence-lab-fzbw9ra6yjwaexwydtdbz4.streamlit.app/)

Explore IPL statistics, team analytics, player performance, match predictions and ML-powered live win probability.

**Tech Stack:** Python | Pandas | SQL | XGBoost | Scikit-learn | Streamlit | Plotly
:::

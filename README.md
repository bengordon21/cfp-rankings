# Modeling and forecasting the CFP committee rankings (2026)

Learn which team attributes explain the committee's weekly Top 25, and forecast each release before it airs.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # then put your CollegeFootballData key in .env (never commit it)
```

## Weekly cycle
```bash
.venv/bin/python run_weekly.py baseline            # one time: Week 10 baseline -> models/week_10_champion.pkl
.venv/bin/python run_weekly.py predict --week 11   # Sun/Mon, after Saturday games, BEFORE Tuesday-night games
.venv/bin/python run_weekly.py score   --week 11   # Tue night: scores the frozen prediction, then runs update
.venv/bin/python run_weekly.py score   --week 11 --csv top25.csv   # manual Top 25 (columns: rank,team)
.venv/bin/python run_weekly.py log --message "dropped X" --reason "why"   # manual tweak -> CHANGELOG.md
```
- `predict` writes `predictions/week_N_prediction.{csv,md}` (timestamped, model version stamped). It refuses to overwrite one; `--redo` archives the old file and is blocked once the week has been scored.
- `score` compares to the committee poll ("Playoff Committee Rankings") plus the baselines: SP+ order, last week's committee order, and AP order (comparison only), then appends to `results/scorecard.csv` and `biggest_misses.csv`.
- `update` adds the week to the training data, re-evaluates every candidate (leave-one-team-out with one week, leave-one-week-out after), chooses the next champion, and refreshes `results/*`, `REPORT.md`, and `CHANGELOG.md`.

## Explanations and app
`update` (and so `baseline` and `score`) also fits a separate ridge explainer on the committee's *actual* Top 25 and writes `explanations/week_N_explanation.md`, `week_N_team_contributions.csv`, `week_N_factor_weights.csv` and `week_N_fit_quality.csv`. The forecasting champion and `predictions/` are untouched.
```bash
.venv/bin/python run_weekly.py explain --week 10   # regenerate one week's files (no API, no retrain)
.venv/bin/streamlit run app.py                     # read-only explorer
```
Run the tests with `.venv/bin/python -m pytest`.

## Files
| File | Role |
|---|---|
| explain.py | Committee explainer: factor weights, per-team contributions, pairs, movers |
| app.py | Streamlit explorer (read-only) |
| fetch.py | CFBD client: disk cache (`data/raw/*.json`), call log (`logs/api_calls.csv`) |
| build_features.py | Feature construction (see FEATURES.md) |
| models.py | Pointwise / pairwise / GBM models, VIF pruning, CV, bootstrap bands |
| run_weekly.py | predict / score / update / baseline CLI, bracket, storylines, call-budget log |
| report.py | Writes REPORT.md |

## API budget
Free tier: 1,000 calls/month. Per new release: about 6 calls for `predict` (games, 4 ratings, advanced stats) and 2 for `score` (committee poll, AP/Coaches).
Everything is cached, and `logs/call_budget.csv` records planned vs actual calls.
If a rating or poll call fails, nothing is guessed: the command stops.

## Honest limits
See the caveats in REPORT.md: tiny sample, heavy collinearity, snapshot timing of ratings, projected (not actual) conference champions.
`data/raw/` is git-ignored (reproducible from the API, key stays local).

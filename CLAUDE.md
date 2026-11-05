# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A weekly forecaster for the 2026 CFP committee Top 25. It trains on committee rankings as they are released, predicts each upcoming release before it airs, scores the prediction afterward, and retrains. Tests are in `tests/` (pytest); there is no linter. The workflow is the CLI in `run_weekly.py`. README.md and FEATURES.md document the user-facing flow and feature definitions in detail.

## Setup and commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # needs CFBD_API_KEY (CollegeFootballData); never commit it

.venv/bin/python run_weekly.py baseline            # one-time: Week 10 -> models/week_10_champion.pkl
.venv/bin/python run_weekly.py predict --week N    # after Saturday games, before the Tuesday show
.venv/bin/python run_weekly.py score   --week N [--csv top25.csv]   # csv columns: rank,team
.venv/bin/python run_weekly.py log --message "..." --reason "..."   # manual tweak -> CHANGELOG.md
.venv/bin/python run_weekly.py explain --week N     # rewrite explanations/week_N_* (no API, no retrain)
.venv/bin/streamlit run app.py                      # read-only explorer
.venv/bin/python -m pytest                          # tests (single test: pytest tests/test_explain.py::test_name)
```

`predict --redo` archives and replaces a prediction, but is blocked once that week is scored. `score` ends by running `update` (retrain and refresh `results/*`, `REPORT.md`, `CHANGELOG.md`) unless `no_update` is set.

## Architecture

Pipeline: `fetch.py` (API + cache) → `build_features.py` (per-team features) → `models.py` (candidate models, CV, champion) → `run_weekly.py` (orchestration, artifacts) → `report.py` (REPORT.md).

- **Release week vs. CFBD week.** Committee "Week N" uses only games from CFBD weeks <= N-1. Features, API calls (`endWeek=N-1`), and stored ratings are all keyed to this cutoff. Keep the off-by-one in mind when touching any of them.
- **Ratings are snapshots.** The CFBD ratings endpoints have no "as of week" option, so each is cached under a release stamp (`*_rel{N}.json`) when `predict` runs and can't be rebuilt later. Files named `*_contaminated` or `*_unused` in `data/raw/` are deliberately retained as non-inputs.
- **Features** are z-scored within each release week across all FBS teams. The three framings (resume / power / combined, plus `combined_inertia` once Week 11 exists) are defined in `models.py`, then pruned for collinearity by VIF > 10 using the fixed `DROP_PRIORITY` order (committee-style résumé items kept, duplicate ratings dropped first). Dropped features are logged to `results/collinearity.csv`.
- **Models** (`models.py`): pointwise, pairwise (with a head-to-head adjustment that swaps adjacent top-40 teams), and GBM, each over a small hyperparameter grid and the framings. `evaluate_candidates` uses leave-one-team-out CV when only one week of training exists and leave-one-week-out afterward. `choose_champion` picks the best mean score across all past weeks, preferring pairwise within 0.01. Champions are pickled as `models/week_N_champion.pkl`; `predict` loads the latest one via `latest_champion_path`.
- **Explainer** (`explain.py`, UI-free): a ridge fit on the committee's *actual* Top 25, separate from the forecasting champion, so each team's score splits exactly into factor contributions. `cmd_update` calls it (wrapped in try/except so it never blocks retraining) and writes `explanations/week_N_*`. In `run_weekly.py` it is imported as `explainer` because that module defines its own `explain()` helper. `app.py` only reads those files plus existing `results/`, `predictions/` and `data/features/`; it must never call the API or write project files. Weights on collinear stats can have misleading signs (see the caveats in each week's explanation).
- **Training data** is `data/training/actuals.csv` (committee ranks by week), appended by `score`.
- **Frozen predictions.** `predictions/week_N_prediction.{csv,md}` are timestamped and stamped with the model version. They must not be altered after the fact, since `score` evaluates them against the committee, SP+ order, last week's committee order, and AP.

## API budget and caching

The free tier allows 1,000 calls/month. Every call goes through `fetch.get`, which caches to `data/raw/*.json` (git-ignored) and logs to `logs/api_calls.csv`. `logs/call_budget.csv` records planned vs. actual calls per command (about 6 for `predict`, 2 for `score`). If a ratings or poll call fails, the command stops; do not substitute guessed data. Check the cache before adding any new endpoint use.

## Conventions

- Every model, feature, or process change is recorded in `CHANGELOG.md` (automatically on retrain, via `log` for manual changes, tagged **[manual]**). Use `run_weekly.py log` rather than editing it by hand.
- `.env` holds the API key and is git-ignored. The directory is not currently a git repository.

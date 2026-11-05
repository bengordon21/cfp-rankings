# Explainability layer and Streamlit app: design

Date: 2026-11-05

## Goal

Each week, show which factors drive the CFP committee's Top 25 and why each team sits where it does. Results are browsable in a Streamlit app. The existing forecasting pipeline (predict / score / update) is unchanged.

## Decisions

- **What is explained:** the committee's actual released ranking, refit after release. The frozen prediction is not touched. The report also notes where the committee diverged from the model.
- **Layers:** league-wide factor weights, plus a per-team breakdown.
- **Delivery:** a Streamlit app (view and explore only), plus markdown and CSV files written by the CLI.
- **App scope:** read-only. It never calls the CFBD API and never writes to `predictions/`, `models/`, `data/` or `results/`. `predict` and `score` stay in the terminal so the API budget and the frozen-prediction rules cannot be broken from the UI.

## Approach: separate linear explainer

After `score` saves the actual ranks, fit a ridge model on those ranks over the kept features of the combined framing. Because the model is linear on within-week z-scored features, each team's score splits exactly into factor x weight contributions. The forecasting champion (pairwise, pointwise or GBM) is not used for explanation, so selection logic and forecast accuracy are unaffected.

Rejected: SHAP on the champion (noisy and unstable on about 25 teams, and it approximates the model, not the committee), and forcing a linear champion (costs forecast accuracy and changes selection logic).

## Components

### `explain.py` (new, no UI code)

Pure functions returning DataFrames.

- `fit_explainer(week)`: pointwise ridge, committee rank to score, on the combined framing's kept features for the week. Also a pooled fit across all scored weeks. The head-to-head adjustment is applied as a separate named factor (`h2h`).
- `fit_quality(...)`: Kendall tau and within-Top-25 pairwise accuracy of the explainer against the real order.
- `factor_weights(...)`: standardized weight per factor, with bootstrap bands (200 resamples, same approach as `bootstrap_ranks`). Factors are grouped as resume, strength of schedule, power ratings, efficiency and context, with each group's share of explained variance. Weights are compared with last week and with the pooled fit.
- `team_contributions(...)`: for each ranked team, the signed contribution of each factor relative to the average ranked team. Contributions plus the baseline sum to the team's explainer score.
- `pair_explanation(a, b)`: the factors separating two teams, plus the h2h result.
- `unexplained(...)`: teams where the committee rank diverges from the explainer rank by more than a threshold, flagged for review.
- `movers(week)`: biggest rank changes versus last week's committee Top 25 and each mover's largest changed factor.

Collinear features dropped by VIF cannot be separated from their kept proxies. The output states this.

### CLI integration (`run_weekly.py`)

`cmd_score` calls the explainer after saving actuals and before `update`. It writes:

- `explanations/week_N_explanation.md`
- `explanations/week_N_team_contributions.csv`
- `explanations/week_N_factor_weights.csv`

`report.py` adds a "What the committee valued" section to REPORT.md. CHANGELOG.md gets a one-line entry. No new API calls: it uses cached features and the committee ranks already fetched.

### `app.py` (new, Streamlit)

Run with `.venv/bin/streamlit run app.py`. `streamlit` and a charting library (Plotly or Altair) are added to `requirements.txt`. A sidebar week selector drives every page. File reads are cached, and a Refresh button reloads them after `score` is run in the terminal.

| Page | Contents |
|---|---|
| Overview | The week's Top 25 (actual vs. predicted, with a rank-change column), plus Kendall tau, within-Top-25 pairwise accuracy and the explainer's fit. Biggest-movers table. |
| What the committee valued | Factor weights with bootstrap bands. Toggle between grouped and individual factors. Line chart of each factor's weight across weeks. |
| Team explorer | Pick a team. Contribution waterfall, raw feature values with z-scores, head-to-head results, and the model's rank band next to the committee rank. |
| Compare two teams | Pick any two teams. The factors separating them and the h2h result. |
| Prediction scorecard | Frozen predictions vs. actuals, baselines (SP+, last week's committee order, AP), biggest misses, and champion history from `results/*.csv`. |
| Data browser | Sortable and filterable feature table for all FBS teams, the collinearity and dropped-feature log, feature drift, API call budget, and CHANGELOG. CSV download on each table. |

If a week has no explanation yet (for example Week 11 before `score`), the page says so and shows only the prediction.

## Error handling

- Missing files or weeks: the app shows a clear message naming the CLI command that produces them. It does not fall back to guessed data.
- Explainer fit on a week with fewer than 25 ranked teams or missing features: the CLI stops with an error and does not write partial explanation files. `score` itself still completes and `update` still runs.
- One week of data is about 25 ranked teams. The markdown and the app display the bootstrap bands and a note that weights are noisy until more weeks are scored.

## Testing

- Unit tests for `explain.py`: contributions plus baseline equal each team's score; fit quality on the Week 10 data is reproducible; bootstrap output has the expected shape.
- A test that running the explainer leaves every file in `predictions/` and `models/` byte-identical.
- App: run it against the Week 10 data and click through each page. No UI test framework.

## Out of scope

- Running predict / score / update from the app.
- What-if tools (editing features or weights).
- Changing the forecasting champion or its selection logic.
- Interactive HTML, custom-designed UI beyond Streamlit theming (`.streamlit/config.toml`).

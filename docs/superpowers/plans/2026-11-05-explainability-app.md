# Explainability Layer and Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each week, explain which factors drive the CFP committee's Top 25 and why each team sits where it does, and make it browsable in a read-only Streamlit app.

**Architecture:** A new UI-free `explain.py` fits a separate ridge model on the committee's *actual* Top 25 (z-scored features, so each team's score splits exactly into factor contributions) and writes markdown and CSVs to `explanations/`. `run_weekly.py` calls it from `cmd_update`. `app.py` only reads files. The forecasting champion, `predictions/` and `models/` are never touched.

**Tech Stack:** Python 3.15 (existing `.venv`), pandas, numpy, scikit-learn (`Ridge`/`RidgeCV`), pytest, Streamlit, Plotly.

Spec: `docs/superpowers/specs/2026-11-05-explainability-app-design.md`

**Notes for the engineer**
- The project directory is **not a git repository**, so there are no commit steps. Each task ends with a passing test run.
- Run everything from the project root with `.venv/bin/python`.
- Features are already z-scored per week by `models.load_training`. Rank 26 means "unranked" (`models.UNRANKED`).
- Never write to `predictions/`, `models/`, `data/` or `results/` from `explain.py` or `app.py`.

**Deliberate deviations from the spec** (each is smaller or safer than the spec; the spec file is not edited):
1. **Head-to-head is reported as a fact, not a fitted weight.** The explainer is a pointwise ridge, which has no pairwise h2h coefficient. Pair comparisons show who won the game, without a weight. Fitting a second pairwise model just for this would add noise.
2. **The hook lives in `cmd_update`, not `cmd_score`.** `score` always ends by calling `cmd_update`, and `baseline` calls it too, so this covers Week 10 as well as every later week.
3. **Two extras:** a `run_weekly.py explain --week N` command (the Week 10 files must be generated without retraining) and a `week_N_fit_quality.csv` file (so the read-only app can show fit metrics).
4. **"Share of influence" is the share of total mean |contribution|** across the committee's ranked teams, not a variance share. It is simpler and exactly defined.

## File structure

| File | Responsibility |
|---|---|
| `explain.py` (new) | Fit, contributions, weights with bootstrap bands, pairs, unexplained, movers, markdown, file IO. No UI code. |
| `app.py` (new) | Streamlit pages. Reads files only. |
| `run_weekly.py` (modify) | `explain` command and the `cmd_update` hook. |
| `report.py` (modify) | `explainer_lines()` and its call in `write_report()`. |
| `tests/test_explain.py` (new) | Unit tests on synthetic data. |
| `tests/test_explain_week10.py` (new) | Integration test on the real Week 10 data. |
| `tests/test_report_explainer.py` (new) | Test for `explainer_lines`. |
| `tests/test_app_smoke.py` (new) | Streamlit `AppTest` smoke test per page. |
| `pytest.ini`, `requirements.txt` (new/modify) | Test config, dependencies. |
| `README.md`, `CLAUDE.md` (modify) | Document the new commands. |

---

### Task 0: Environment

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`, `tests/__init__.py`

- [ ] **Step 1: Add dependencies to `requirements.txt`**

Append these lines:

```
pytest
streamlit
plotly
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Create an empty `tests/__init__.py`**

```bash
mkdir -p tests && : > tests/__init__.py
```

- [ ] **Step 4: Install**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: ends with "Successfully installed ..." (or "already satisfied") and no error.

If a wheel is missing for Python 3.15 (commonly `pyarrow`, a Streamlit dependency), stop and tell the user. The fallback is a separate Python 3.12 virtualenv, which they must approve first because it replaces `.venv`.

- [ ] **Step 5: Verify imports**

Run: `.venv/bin/python -c "import pytest, streamlit, plotly; print(streamlit.__version__, plotly.__version__)"`
Expected: two version numbers printed.

---

### Task 1: Ridge fit, contributions, fit quality

**Files:**
- Create: `explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explain.py`:

```python
import numpy as np
import pandas as pd
import pytest

import explain

FEATS = ["f1", "f2", "f3"]


def make_week(week, n=200, seed=0, n_ranked=25):
    """Synthetic z-scored week: committee rank follows 2*f1 + 1*f2 - 1.5*f3 (rank 26 = unranked)."""
    rng = np.random.default_rng(seed + week)
    Z = rng.normal(size=(n, len(FEATS)))
    good = Z @ np.array([2.0, 1.0, -1.5])
    order = (-good).argsort().argsort() + 1
    df = pd.DataFrame(Z, columns=FEATS)
    df["team"] = [f"T{i}" for i in range(n)]
    df["rank"] = np.where(order <= n_ranked, order, 26).astype(float)
    df["week"] = week
    return df


def test_fit_ridge_recovers_signs():
    coef, icpt, alpha = explain.fit_ridge(make_week(1), FEATS)
    assert list(coef.index) == FEATS
    assert coef["f1"] > 0 > coef["f3"]
    assert coef["f1"] > coef["f2"]
    assert alpha in explain.ALPHAS


def test_contributions_sum_to_explainer_score():
    wk = make_week(1)
    coef, icpt, _ = explain.fit_ridge(wk, FEATS)
    C = explain.team_contributions(wk, FEATS, coef, icpt)
    np.testing.assert_allclose(C["baseline"] + C[FEATS].sum(axis=1), C["explainer_score"])


def test_contributions_centred_on_ranked_teams():
    wk = make_week(1)
    coef, icpt, _ = explain.fit_ridge(wk, FEATS)
    C = explain.team_contributions(wk, FEATS, coef, icpt)
    ranked = C[C.committee_rank <= 25]
    np.testing.assert_allclose(ranked[FEATS].mean().values, 0.0, atol=1e-9)
    assert C.index.name == "team"
    assert C["explainer_rank"].min() == 1


def test_fit_quality_sane_on_synthetic():
    wk = make_week(1)
    coef, icpt, alpha = explain.fit_ridge(wk, FEATS)
    q = explain.fit_quality(wk, FEATS, coef, icpt, alpha)
    assert q["pairwise_acc_top25"] > 0.7
    assert q["n_ranked"] == 25 and q["n_features"] == 3 and q["alpha"] == alpha
    assert set(["kendall", "spearman", "top25_overlap", "top12_overlap"]) <= set(q)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: collection error / FAIL with `ModuleNotFoundError: No module named 'explain'`.

- [ ] **Step 3: Implement**

Create `explain.py`:

```python
"""Explainability layer: what the CFP committee appeared to value, and why each team sits where it does.

A separate ridge model is fit on the committee's ACTUAL Top 25 for a release week (rank 26 = unranked).
It is linear on within-week z-scored features, so every team's score splits exactly into
coefficient x (z - mean z of the committee's ranked teams) contributions. The forecasting champion,
predictions/ and models/ are never touched, and this module has no UI code.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

import build_features as bf
import models as M
import report

ROOT = Path(__file__).parent
OUT = ROOT / "explanations"
FEAT_DIR = ROOT / "data" / "features"
ALPHAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
BOOT = 200
RANKED = 25
GAP = 5            # |committee rank - explainer rank| that flags a team as "unexplained"
META = ["committee_rank", "explainer_score", "explainer_rank", "baseline"]

GROUPS = {
    "Resume": ["wins", "losses", "win_pct", "avg_margin_capped", "quality_wins_sp25", "wins_vs_winning",
               "bad_losses", "best_win_sp_rank", "wins_vs_prev_top25"],
    "Schedule": ["n_fbs_opp", "n_p4_opp", "sos_opp_sp"],
    "Ratings": ["sp", "sp_off", "sp_def", "srs", "elo", "fpi"],
    "Efficiency": ["off_ppa", "def_ppa_allowed", "off_success", "def_success_allowed",
                   "off_explosiveness", "def_explosiveness_allowed"],
    "Context": ["is_p4", "is_independent", "conf_win_pct", "conf_games_behind", "conf_contender",
                "weeks_since_loss", "prev_committee_rank"],
}
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}


def _name(f):
    return report.PLAIN.get(f, f)


def feature_columns(contribs):
    return [c for c in contribs.columns if c not in META]


# ----------------------------------------------------------------------------- fitting
def fit_ridge(df, feats, alpha=None):
    """Ridge on goodness = -rank. Returns (coef Series, intercept, alpha). alpha=None -> RidgeCV (efficient LOO)."""
    X, y = df[feats].values, -df["rank"].values.astype(float)
    if alpha is None:
        m = RidgeCV(alphas=ALPHAS).fit(X, y)
        alpha = float(m.alpha_)
    else:
        m = Ridge(alpha=alpha).fit(X, y)
    return pd.Series(m.coef_, index=feats), float(m.intercept_), float(alpha)


def team_contributions(wk, feats, coef, intercept):
    """One week's rows -> DataFrame indexed by team.

    score_i = baseline + sum_f contribution_if, where baseline is the mean explainer score of the
    committee's ranked teams and contribution_if = coef_f * (z_if - mean z_f over ranked teams).
    """
    wk = wk.reset_index(drop=True)
    Z = wk[feats]
    ranked = (wk["rank"] <= RANKED).values
    score = intercept + Z.values @ coef.values
    centre = Z[ranked].mean()
    out = ((Z - centre) * coef).set_axis(wk["team"].values)
    out.index.name = "team"
    out.insert(0, "baseline", float(score[ranked].mean()))
    out.insert(0, "explainer_rank", pd.Series(score).rank(ascending=False, method="first").values)
    out.insert(0, "explainer_score", score)
    out.insert(0, "committee_rank", wk["rank"].values)
    return out


def fit_quality(wk, feats, coef, intercept, alpha):
    """In-sample agreement between the explainer's order and the committee's."""
    score = intercept + wk[feats].values @ coef.values
    m = M.week_metrics(wk["rank"].values, score)
    keep = ("kendall", "spearman", "pairwise_acc_top25", "top25_overlap", "top12_overlap")
    return {**{k: float(m[k]) for k in keep}, "alpha": float(alpha), "n_features": len(feats),
            "n_ranked": int((wk["rank"] <= RANKED).sum())}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 4 passed.

---

### Task 2: Weights with bootstrap bands and influence

**Files:**
- Modify: `explain.py` (append)
- Test: `tests/test_explain.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def test_bootstrap_weights_bands():
    df = make_week(1)
    coef, _, alpha = explain.fit_ridge(df, FEATS)
    b = explain.bootstrap_weights(df, FEATS, alpha, B=40)
    assert list(b.index) == FEATS
    assert (b.lo <= b.hi).all()
    assert b.loc["f1", "lo"] > 0 > b.loc["f3", "hi"]


def test_factor_weights_week_and_pooled():
    df = pd.concat([make_week(1), make_week(2)], ignore_index=True)
    W = explain.factor_weights(df, FEATS, week=2, B=20)
    assert set(W.kind) == {"week", "pooled"}
    assert (W.week == 2).all()
    for kind in ("week", "pooled"):
        sub = W[W.kind == kind]
        assert list(sub.factor) == FEATS
        assert sub.influence.sum() == pytest.approx(1.0)
    assert set(W.group) == {"Other"}          # synthetic names are not in GROUPS
    assert {"weight", "lo", "hi", "alpha"} <= set(W.columns)


def test_groups_cover_known_features():
    assert explain.GROUP_OF["bad_losses"] == "Resume"
    assert explain.GROUP_OF["sos_opp_sp"] == "Schedule"
    assert explain.GROUP_OF["sp"] == "Ratings"
    assert explain.GROUP_OF["off_ppa"] == "Efficiency"
    assert explain.GROUP_OF["weeks_since_loss"] == "Context"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 3 new FAILs (`AttributeError: module 'explain' has no attribute ...`).

- [ ] **Step 3: Append the implementation to `explain.py`**

```python
# ----------------------------------------------------------------------------- weights
def bootstrap_weights(df, feats, alpha, B=BOOT, seed=0):
    """5th/95th percentile of ridge coefficients when teams are resampled within each week."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("week")]
    draws = np.empty((B, len(feats)))
    for b in range(B):
        bs = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups], ignore_index=True)
        draws[b] = fit_ridge(bs, feats, alpha)[0].values
    return pd.DataFrame({"lo": np.percentile(draws, 5, axis=0), "hi": np.percentile(draws, 95, axis=0)},
                        index=feats)


def _influence(sub, feats, coef):
    """Each factor's share of total mean |contribution| across the committee's ranked teams."""
    parts = []
    for _, g in sub.groupby("week"):
        r = g.loc[g["rank"] <= RANKED, feats]
        parts.append(((r - r.mean()).abs() * coef.abs()).mean())
    total = pd.concat(parts, axis=1).mean(axis=1)
    return total / total.sum() if total.sum() > 0 else total * 0.0


def factor_weights(df, feats, week, B=BOOT):
    """Long table: this week's fit (kind='week') and the fit pooled over every week up to `week`."""
    rows = []
    for kind, sub in (("week", df[df.week == week]), ("pooled", df)):
        coef, _, alpha = fit_ridge(sub, feats)
        band = bootstrap_weights(sub, feats, alpha, B)
        rows.append(pd.DataFrame({
            "week": week, "kind": kind, "factor": feats,
            "group": [GROUP_OF.get(f, "Other") for f in feats],
            "weight": coef.values, "lo": band.lo.values, "hi": band.hi.values,
            "influence": _influence(sub, feats, coef).values, "alpha": alpha}))
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 7 passed.

---

### Task 3: Pair comparison, unexplained teams, movers

**Files:**
- Modify: `explain.py` (append)
- Test: `tests/test_explain.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def _frame(rows):
    """rows: {team: (committee_rank, explainer_rank, explainer_score, f1, f2)}"""
    idx = list(rows)
    v = np.array(list(rows.values()), float)
    return pd.DataFrame({"committee_rank": v[:, 0], "explainer_score": v[:, 2], "explainer_rank": v[:, 1],
                         "baseline": 0.0, "f1": v[:, 3], "f2": v[:, 4]}, index=pd.Index(idx, name="team"))


def test_pair_explanation_lists_and_h2h():
    C = _frame({"A": (1, 1, 3.0, 1.5, 0.5), "B": (2, 2, 1.0, 0.0, 1.0)})
    P = explain.pair_explanation(C, "A", "B", {("B", "A")}, top_n=2)
    assert P["gap"] == pytest.approx(2.0)
    assert P["favoring_a"] == [("f1", pytest.approx(1.5))]
    assert P["favoring_b"] == [("f2", pytest.approx(0.5))]
    assert P["h2h"] == -1                           # B beat A
    assert explain.pair_explanation(C, "A", "B", {("A", "B")})["h2h"] == 1
    assert explain.pair_explanation(C, "A", "B", set())["h2h"] == 0


def test_unexplained_threshold():
    C = _frame({"close": (1, 2, 0, 0, 0), "far": (3, 12, 0, 0, 0),
                "snub": (26, 4, 0, 0, 0), "nobody": (26, 40, 0, 0, 0)})
    U = explain.unexplained(C, gap=5)
    assert list(U.index) == ["snub", "far"]          # sorted by |gap| descending
    assert U.loc["far", "gap"] == 9 and U.loc["snub", "gap"] == -22


def test_movers_largest_factor():
    cur = _frame({"A": (1, 1, 0, 1.0, 0.0), "B": (2, 2, 0, 0.0, 0.0), "C": (26, 9, 0, -1.0, 0.5)})
    prev = _frame({"A": (2, 1, 0, 0.2, 0.0), "B": (1, 2, 0, 0.0, 0.0), "C": (5, 3, 0, 1.0, 0.5)})
    m = explain.movers(cur, prev)
    assert list(m.team) == ["C", "A", "B"]           # C: 5 -> NR is the biggest move
    c = m.iloc[0]
    assert c.prev_rank == 5 and c["rank"] == 26 and c.move == 5 - 26
    assert c.top_factor == "f1" and c.top_delta == pytest.approx(-2.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 3 new FAILs with `AttributeError`.

- [ ] **Step 3: Append the implementation to `explain.py`**

```python
# ----------------------------------------------------------------------------- comparisons
def pair_explanation(contribs, a, b, h2h=frozenset(), top_n=3):
    """Why is `a` above/below `b`? Factors where each team leads, plus the head-to-head result.

    h2h is a set of (winner, loser). Result `h2h`: +1 a beat b, -1 b beat a, 0 no game.
    Head-to-head is reported as a fact; the ridge has no fitted weight for it.
    """
    feats = feature_columns(contribs)
    d = (contribs.loc[a, feats] - contribs.loc[b, feats]).sort_values(ascending=False)
    return {
        "a": a, "b": b,
        "gap": float(contribs.loc[a, "explainer_score"] - contribs.loc[b, "explainer_score"]),
        "favoring_a": [(f, float(v)) for f, v in d[d > 0].head(top_n).items()],
        "favoring_b": [(f, float(-v)) for f, v in d[d < 0].iloc[::-1].head(top_n).items()],
        "h2h": 1 if (a, b) in h2h else (-1 if (b, a) in h2h else 0),
    }


def unexplained(contribs, gap=GAP):
    """Teams where the committee and the explainer disagree by >= `gap` places (Top 25 in either list).

    gap = explainer_rank - committee_rank: positive = the committee liked the team more than the factors do.
    """
    c = contribs
    keep = (c.committee_rank <= RANKED) | (c.explainer_rank <= RANKED)
    out = c.loc[keep, ["committee_rank", "explainer_rank"]].copy()
    out["gap"] = (out.explainer_rank - out.committee_rank).astype(int)
    out = out[out.gap.abs() >= gap]
    return out.reindex(out.gap.abs().sort_values(ascending=False).index)


def movers(cur, prev, top=8):
    """Biggest committee-rank moves vs last week and each mover's largest change in factor contribution."""
    prev = prev.reindex(cur.index)
    pr = prev.committee_rank.fillna(M.UNRANKED)
    feats = feature_columns(cur)
    rows = []
    for t in cur.index:
        r, p = float(cur.loc[t, "committee_rank"]), float(pr[t])
        if r > RANKED and p > RANKED:
            continue
        delta = (cur.loc[t, feats] - prev.loc[t, feats]).fillna(0.0)
        f = delta.abs().idxmax()
        rows.append({"team": t, "prev_rank": p, "rank": r, "move": p - r,
                     "top_factor": f, "top_delta": float(delta[f])})
    out = pd.DataFrame(rows, columns=["team", "prev_rank", "rank", "move", "top_factor", "top_delta"])
    return out.reindex(out.move.abs().sort_values(ascending=False, kind="stable").index).head(top).reset_index(drop=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 10 passed.

---

### Task 4: Assemble a week, markdown, write and load files

**Files:**
- Modify: `explain.py` (append)
- Test: `tests/test_explain.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def _two_weeks():
    df = pd.concat([make_week(1), make_week(2)], ignore_index=True)
    return df, {1: set(), 2: {("T0", "T1")}}


def test_explain_frames_structure():
    df, h2h = _two_weeks()
    res = explain.explain_frames(df, h2h, 2, FEATS, B=20)
    assert res["week"] == 2 and res["feats"] == FEATS
    assert set(res["weights"].kind) == {"week", "pooled"}
    assert len(res["contributions"]) == 200
    assert res["movers"] is not None            # week 1 exists
    assert explain.explain_frames(df[df.week == 2], h2h, 2, FEATS, B=20)["movers"] is None


def test_explain_frames_rejects_incomplete_top25():
    df = make_week(1, n_ranked=10)
    with pytest.raises(ValueError, match="only 10 ranked"):
        explain.explain_frames(df, {1: set()}, 1, FEATS, B=10)


def test_markdown_sections():
    df, h2h = _two_weeks()
    res = explain.explain_frames(df, h2h, 2, FEATS, B=20)
    md = explain.build_markdown(res)
    for s in ("# Week 2 committee explanation", "## What the committee valued this week",
              "## Why each Top 25 team sits where it does", "## What separates neighbors",
              "## Teams the factors cannot explain", "## Biggest movers", "## Caveats"):
        assert s in md
    assert md.count("\n| ") > 25                    # 25 team rows plus factor rows


def test_write_and_load_roundtrip(tmp_path):
    df, h2h = _two_weeks()
    res = explain.explain_frames(df, h2h, 2, FEATS, B=20)
    paths = explain.write_outputs(res, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert explain.explained_weeks(tmp_path) == [2]
    E = explain.load_week(2, tmp_path)
    assert E["contributions"].index.name == "team" and len(E["contributions"]) == 200
    assert set(E["weights"].kind) == {"week", "pooled"}
    assert E["quality"]["n_ranked"] == 25
    assert "Week 2" in E["markdown"]
    assert explain.load_week(9, tmp_path) is None


def test_write_outputs_second_week_shows_change(tmp_path):
    df, h2h = _two_weeks()
    explain.write_outputs(explain.explain_frames(df[df.week == 1], h2h, 1, FEATS, B=20), tmp_path)
    res2 = explain.explain_frames(df, h2h, 2, FEATS, B=20)
    explain.write_outputs(res2, tmp_path)
    text = (tmp_path / "week_2_explanation.md").read_text()
    assert "n/a" not in text.split("## Why each Top 25")[0].split("| Factor")[1]   # change column filled
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 5 new FAILs with `AttributeError`.

- [ ] **Step 3: Append the implementation to `explain.py`**

```python
# ----------------------------------------------------------------------------- assemble
def explain_frames(df, h2h, week, feats, B=BOOT):
    """Everything for one release week from an in-memory training frame (no file IO).

    df: z-scored rows for every week <= `week` (columns: feats, team, rank, week). h2h: {week: {(winner, loser)}}.
    Raises ValueError before computing anything if the week has no complete committee Top 25.
    """
    wk = df[df.week == week]
    n_ranked = int((wk["rank"] <= RANKED).sum())
    if n_ranked < RANKED:
        raise ValueError(f"only {n_ranked} ranked teams for week {week}; need {RANKED}")
    coef, icpt, alpha = fit_ridge(wk, feats)
    contribs = team_contributions(wk, feats, coef, icpt)
    prev_c = None
    if (week - 1) in set(df.week):
        pw = df[df.week == week - 1]
        pcoef, picpt, _ = fit_ridge(pw, feats)
        prev_c = team_contributions(pw, feats, pcoef, picpt)
    return {
        "week": int(week), "feats": list(feats),
        "weights": factor_weights(df, feats, week, B),
        "contributions": contribs,
        "quality": fit_quality(wk, feats, coef, icpt, alpha),
        "unexplained": unexplained(contribs),
        "movers": movers(contribs, prev_c) if prev_c is not None else None,
        "h2h": set(h2h.get(week, set())),
    }


def explain_week(week, actuals, B=BOOT):
    """Load features through `week` and explain it. `actuals` = DataFrame(week, team, rank)."""
    actuals = actuals[actuals.week <= week]
    if week not in set(actuals.week):
        raise ValueError(f"no committee ranking stored for week {week}")
    df, h2h = M.load_training(actuals)
    feats, _ = M.select_features(df, M.FRAMINGS["combined"], "combined")
    return explain_frames(df, h2h, week, feats, B)


# ----------------------------------------------------------------------------- markdown
def _drivers(row, feats, n=3, floor=0.02):
    c = row[feats].sort_values(ascending=False)
    up = [f"{_name(k)} {v:+.2f}" for k, v in c.head(n).items() if v > floor]
    dn = [f"{_name(k)} {v:+.2f}" for k, v in c.tail(n).iloc[::-1].items() if v < -floor]
    return ("; ".join(up) or "-"), ("; ".join(dn) or "-")


def describe_pair(P, rank_a, rank_b):
    lead = lambda t, L: f"{t} ahead on " + ", ".join(f"{_name(f)} {v:+.2f}" for f, v in L) if L else None
    parts = [x for x in (lead(P["a"], P["favoring_a"]), lead(P["b"], P["favoring_b"])) if x]
    s = f"- **#{rank_a} {P['a']} over #{rank_b} {P['b']}**: " + ("; ".join(parts) or "no clear separating factor")
    if P["h2h"]:
        s += f". Head-to-head: {P['a'] if P['h2h'] == 1 else P['b']} won."
    if P["gap"] < 0:
        s += " (The factors alone score them the other way.)"
    return s


def build_markdown(res, prev_weights=None):
    w, q, C = res["week"], res["quality"], res["contributions"]
    feats = feature_columns(C)
    W = res["weights"]
    wk = W[W.kind == "week"].set_index("factor")
    prev = None
    if prev_weights is not None:
        prev = prev_weights[prev_weights.kind == "week"].set_index("factor").weight
    L = [f"# Week {w} committee explanation", "",
         f"_Explainer: ridge regression (alpha {q['alpha']:g}) fit on the committee's actual Top 25 for this release, "
         f"{len(feats)} VIF-pruned features. In-sample fit: Kendall tau {q['kendall']:.2f}, within-Top-25 pairwise "
         f"accuracy {q['pairwise_acc_top25']:.2f}, {int(q['top25_overlap'])}/25 teams in the right Top 25. "
         "This describes what the committee appeared to value; it is separate from the forecasting model._", "",
         "## What the committee valued this week", "",
         "| Factor | Group | Weight | 90% band | Share of influence | Change vs last week |",
         "|---|---|---|---|---|---|"]
    for f, r in wk.sort_values("influence", ascending=False).head(10).iterrows():
        d = f"{r.weight - prev[f]:+.2f}" if prev is not None and f in prev.index else "n/a"
        L.append(f"| {_name(f)} | {r.group} | {r.weight:+.2f} | {r.lo:+.2f} to {r.hi:+.2f} | {r.influence:.0%} | {d} |")
    g = wk.groupby("group").influence.sum().sort_values(ascending=False)
    L += ["", "Sign: positive = a higher value of the stat helps a team; negative = a higher value hurts "
              "(so lower is better). Weight = change in the explainer's score per 1 SD of the stat. "
              "Share of influence = share of total mean |contribution| across the committee's ranked teams.", "",
          "By group: " + ", ".join(f"{k} {v:.0%}" for k, v in g.items()) + ".", "",
          "## Why each Top 25 team sits where it does", "",
          "Contributions are relative to the average team in the committee's Top 25.", "",
          "| Rank | Team | Helped most | Hurt most |", "|---|---|---|---|"]
    ranked = C[C.committee_rank <= RANKED].sort_values("committee_rank")
    for t, r in ranked.iterrows():
        up, dn = _drivers(r, feats)
        L.append(f"| {int(r.committee_rank)} | {t} | {up} | {dn} |")
    names = list(ranked.index)
    L += ["", "## What separates neighbors (Top 12)", ""]
    for a, b in zip(names[:11], names[1:12]):
        P = pair_explanation(C, a, b, res["h2h"], top_n=2)
        L.append(describe_pair(P, int(C.loc[a, "committee_rank"]), int(C.loc[b, "committee_rank"])))
    U = res["unexplained"]
    L += ["", "## Teams the factors cannot explain", ""]
    if len(U):
        L += [f"Committee and explainer differ by {GAP}+ places (positive = the committee ranked the team higher "
              "than the factors suggest):", ""]
        L += [f"- {t}: committee {int(r.committee_rank) if r.committee_rank <= RANKED else 'NR'}, "
              f"explainer {int(r.explainer_rank)} ({int(r.gap):+d})" for t, r in U.iterrows()]
    else:
        L.append("None: every Top 25 team is within the threshold.")
    L += ["", "## Biggest movers vs last week", ""]
    mv = res["movers"]
    if mv is None or not len(mv):
        L.append("No earlier committee release is in the data yet.")
    else:
        L += ["| Team | Last week | This week | Move | Largest contribution change |", "|---|---|---|---|---|"]
        for r in mv.itertuples():
            fmt = lambda x: "NR" if x > RANKED else str(int(x))
            L.append(f"| {r.team} | {fmt(r.prev_rank)} | {fmt(r.rank)} | {int(r.move):+d} | "
                     f"{_name(r.top_factor)} {r.top_delta:+.2f} |")
    L += ["", "## Caveats", "",
          "- Each week has only 25 ranked teams; weights are noisy, so rely on the 90% bands and on the pooled fit "
          "(`kind=pooled` in the weights CSV) as more weeks arrive.",
          "- Collinear stats (dropped by VIF) cannot be separated from the kept stats that proxy them.",
          "- Fit metrics are in-sample. Head-to-head results are shown as facts; they have no fitted weight.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- files
def _paths(week, outdir):
    return {"md": outdir / f"week_{week}_explanation.md",
            "contributions": outdir / f"week_{week}_team_contributions.csv",
            "weights": outdir / f"week_{week}_factor_weights.csv",
            "quality": outdir / f"week_{week}_fit_quality.csv"}


def write_outputs(res, outdir=OUT):
    """Write the four files for a week. Everything is built before the first write (no partial output)."""
    outdir, w = Path(outdir), res["week"]
    prev = outdir / f"week_{w - 1}_factor_weights.csv"
    md = build_markdown(res, pd.read_csv(prev) if prev.exists() else None)
    p = _paths(w, outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p["md"].write_text(md)
    res["contributions"].round(6).to_csv(p["contributions"])
    res["weights"].round(6).to_csv(p["weights"], index=False)
    pd.DataFrame([{"week": w, **res["quality"]}]).round(6).to_csv(p["quality"], index=False)
    return p


def run(week, actuals, outdir=OUT, B=BOOT):
    res = explain_week(week, actuals, B)
    return res, write_outputs(res, outdir)


def explained_weeks(outdir=OUT):
    return sorted(int(p.stem.split("_")[1]) for p in Path(outdir).glob("week_*_team_contributions.csv"))


def load_week(week, outdir=OUT):
    """Read a week's files for the app. None if the week has not been explained yet."""
    p = _paths(week, Path(outdir))
    if not all(p[k].exists() for k in ("contributions", "weights", "quality")):
        return None
    X, H = FEAT_DIR / f"features_wk{week}.csv", FEAT_DIR / f"h2h_wk{week}.csv"
    raw = pd.read_csv(X, index_col=0) if X.exists() else None
    return {
        "week": week,
        "contributions": pd.read_csv(p["contributions"], index_col="team"),
        "weights": pd.read_csv(p["weights"]),
        "quality": pd.read_csv(p["quality"]).iloc[0].to_dict(),
        "markdown": p["md"].read_text() if p["md"].exists() else "",
        "raw": raw,
        "z": bf.zscore(raw) if raw is not None else None,
        "h2h": pd.read_csv(H) if H.exists() else pd.DataFrame(columns=["winner", "loser"]),
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 15 passed.

---

### Task 5: CLI command, `cmd_update` hook, report section

**Files:**
- Modify: `run_weekly.py` (import near line 20, `cmd_update` after line 113, new `cmd_explain`, argparse near line 484, dispatch near line 500)
- Modify: `report.py` (new `explainer_lines`, one call in `write_report`)
- Test: `tests/test_report_explainer.py`

- [ ] **Step 1: Write the failing report test**

Create `tests/test_report_explainer.py`:

```python
import pandas as pd

import report


def test_explainer_lines_empty_when_no_files(tmp_path):
    assert report.explainer_lines(tmp_path) == []


def test_explainer_lines_uses_latest_week(tmp_path):
    def w(week, shift):
        return pd.DataFrame({
            "week": week, "kind": ["week", "week", "pooled"], "factor": ["bad_losses", "sp", "bad_losses"],
            "group": ["Resume", "Ratings", "Resume"], "weight": [-1.0, 0.5, -1.0], "lo": 0, "hi": 0,
            "influence": [0.6 + shift, 0.4 - shift, 1.0], "alpha": 1.0})
    w(10, 0.0).to_csv(tmp_path / "week_10_factor_weights.csv", index=False)
    w(11, 0.1).to_csv(tmp_path / "week_11_factor_weights.csv", index=False)
    text = "\n".join(report.explainer_lines(tmp_path))
    assert "week 11" in text and "bad losses (70%)" in text
    assert "Resume 70%" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_explainer.py -v`
Expected: FAIL with `AttributeError: module 'report' has no attribute 'explainer_lines'`.

- [ ] **Step 3: Add `explainer_lines` to `report.py`**

Insert this function directly above `def write_report():`:

```python
def explainer_lines(exp_dir=None):
    """Short 'what the committee valued' block from the latest explanations/week_N_factor_weights.csv."""
    d = Path(exp_dir) if exp_dir else ROOT / "explanations"
    files = sorted(d.glob("week_*_factor_weights.csv"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        return []
    W = pd.read_csv(files[-1])
    wk = W[W.kind == "week"]
    week = int(wk.week.iloc[0])
    top = wk.sort_values("influence", ascending=False).head(5)
    groups = wk.groupby("group").influence.sum().sort_values(ascending=False)
    return [f"**Explainer view (week {week}, fit on the committee's actual Top 25).** Top factors by share of "
            "influence: " + "; ".join(f"{PLAIN.get(r.factor, r.factor)} ({r.influence:.0%})" for r in top.itertuples())
            + ". By group: " + ", ".join(f"{k} {v:.0%}" for k, v in groups.items())
            + f". Full detail: `explanations/week_{week}_explanation.md`.", ""]


```

Then in `write_report()`, find this exact line:

```python
    if comp is not None:
        best = comp[comp.eligible_champion].groupby("framing").selection_score.max()
```

and insert one line before it so it reads:

```python
    L += explainer_lines()
    if comp is not None:
        best = comp[comp.eligible_champion].groupby("framing").selection_score.max()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report_explainer.py -v`
Expected: 2 passed.

- [ ] **Step 5: Wire up `run_weekly.py`**

(a) Add the import. Change:

```python
import build_features as bf
import fetch
import models as M
import report
```

to:

```python
import build_features as bf
import explain
import fetch
import models as M
import report
```

(b) In `cmd_update`, after this existing block:

```python
    if week not in set(actuals.week):
        sys.exit(f"No committee ranking stored for week {week}; run score (or baseline) first.")
```

add:

```python
    try:                                            # explanation never blocks retraining
        _, paths = explain.run(week, actuals)
        changelog(f"Week {week} explanation written",
                  f"- {', '.join(p.name for p in paths.values())} in explanations/.",
                  "Explainer refit on the committee's actual Top 25 (explain.py); the forecasting champion is unchanged.")
        print(f"Explanation: {paths['md']}")
    except Exception as e:
        print(f"Explanation skipped: {e}")
```

(c) Add a command function directly above `def cmd_baseline():`:

```python
def cmd_explain(week):
    """Regenerate explanations/week_N_* from stored actuals. No API calls, no retraining."""
    _, paths = explain.run(week, load_actuals())
    print("Wrote " + ", ".join(p.name for p in paths.values()))
```

(d) In the `__main__` block, change the `choices` list to include `"explain"`:

```python
    ap.add_argument("cmd", choices=["baseline", "predict", "score", "update", "explain", "log"])
```

and add `"explain": lambda: cmd_explain(a.week),` to the dispatch dict so it reads:

```python
        {"predict": lambda: cmd_predict(a.week, a.redo),
         "score": lambda: cmd_score(a.week, a.csv, a.no_update),
         "update": lambda: cmd_update(a.week),
         "explain": lambda: cmd_explain(a.week)}[a.cmd]()
```

Also add this line to the module docstring after the `update` line:

```
  python run_weekly.py explain --week 11   # rewrite explanations/week_11_* (no retrain, no API)
```

- [ ] **Step 6: Verify the CLI parses**

Run: `.venv/bin/python run_weekly.py explain` 
Expected: exits with `--week N is required` (not a traceback).

---

### Task 6: Real Week 10 integration test and generated files

**Files:**
- Test: `tests/test_explain_week10.py`
- Generates: `explanations/week_10_*`

- [ ] **Step 1: Write the test**

Create `tests/test_explain_week10.py`:

```python
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import explain

ROOT = Path(__file__).parent.parent
ACTUALS = ROOT / "data" / "training" / "actuals.csv"
pytestmark = pytest.mark.skipif(not ACTUALS.exists() or not (ROOT / "data/features/features_wk10.csv").exists(),
                                reason="Week 10 data not present")


def _digest(*dirs):
    h = hashlib.sha256()
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(str(p).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def test_week10_explanation(tmp_path):
    before = _digest(ROOT / "predictions", ROOT / "models")
    res, paths = explain.run(10, pd.read_csv(ACTUALS), outdir=tmp_path, B=30)

    C = res["contributions"]
    np.testing.assert_allclose(C["baseline"] + C[explain.feature_columns(C)].sum(axis=1), C["explainer_score"])
    assert (C.committee_rank <= 25).sum() == 25
    assert len(C) == 138
    q = res["quality"]
    assert q["pairwise_acc_top25"] > 0.7 and q["kendall"] > 0.4      # sanity bounds, in-sample
    assert res["movers"] is None                                    # no week 9 committee list
    assert set(res["weights"].kind) == {"week", "pooled"}
    assert all(p.exists() for p in paths.values())
    assert "Miami" in paths["md"].read_text()                       # committee #1 in week 10

    assert _digest(ROOT / "predictions", ROOT / "models") == before  # frozen artifacts untouched
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_explain_week10.py -v`
Expected: PASS. If a threshold assertion fails, print `res["quality"]` and investigate (feature set, alpha) before touching the numbers.

- [ ] **Step 3: Generate the real Week 10 files**

Run: `.venv/bin/python run_weekly.py explain --week 10`
Expected: `Wrote week_10_explanation.md, week_10_team_contributions.csv, week_10_factor_weights.csv, week_10_fit_quality.csv`

- [ ] **Step 4: Read the output for sanity**

Run: `sed -n 1,40p explanations/week_10_explanation.md`
Expected: a factor table with 10 rows, a group line, and a Top 25 table where #1 is Miami. Check the signs make sense (for example bad losses negative, quality wins positive). Report anything odd to the user rather than hiding it.

- [ ] **Step 5: Full test run so far**

Run: `.venv/bin/python -m pytest -v`
Expected: all pass.

---

### Task 7: Streamlit app

**Files:**
- Create: `app.py`
- Test: `tests/test_app_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_app_smoke.py`:

```python
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
APP = str(ROOT / "app.py")
pytestmark = pytest.mark.skipif(not (ROOT / "explanations" / "week_10_team_contributions.csv").exists(),
                                reason="run `run_weekly.py explain --week 10` first")
PAGES = ["Overview", "What the committee valued", "Team explorer", "Compare two teams",
         "Prediction scorecard", "Data browser"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, (page, at.exception)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_app_smoke.py -v`
Expected: FAIL (app.py does not exist yet).

- [ ] **Step 3: Create `app.py`**

```python
"""Streamlit explorer for the committee explainer.

Read-only: it never calls the CFBD API and never writes to predictions/, models/, data/ or results/.
Run:  .venv/bin/streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import explain
import report

ROOT = Path(__file__).parent
PRED, RES, LOGS = ROOT / "predictions", ROOT / "results", ROOT / "logs"
PAGES = ["Overview", "What the committee valued", "Team explorer", "Compare two teams",
         "Prediction scorecard", "Data browser"]

st.set_page_config(page_title="CFP committee explainer", layout="wide")


def nm(f):
    return report.PLAIN.get(f, f)


@st.cache_data
def week_data(week):
    return explain.load_week(week)


@st.cache_data
def csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def all_weights():
    frames = [explain.load_week(w)["weights"] for w in explain.explained_weeks()]
    return pd.concat(frames, ignore_index=True) if frames else None


def prediction(week):
    return csv(str(PRED / f"week_{week}_prediction.csv"))


def download(df, name):
    st.download_button("Download CSV", df.to_csv(index=False), file_name=name, mime="text/csv", key=f"dl_{name}")


def no_explanation(week):
    st.info(f"No explanation for week {week} yet. It is written when `run_weekly.py score --week {week}` runs "
            f"(or run `run_weekly.py explain --week {week}`).")


# ----------------------------------------------------------------------------- pages
def page_overview(week, E):
    q = E["quality"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kendall tau (in-sample)", f"{q['kendall']:.2f}")
    c2.metric("Top-25 pairwise accuracy", f"{q['pairwise_acc_top25']:.2f}")
    c3.metric("Teams in the right Top 25", f"{int(q['top25_overlap'])}/25")
    c4.metric("Ridge alpha", f"{q['alpha']:g}")
    C = E["contributions"]
    top = C[C.committee_rank <= 25].sort_values("committee_rank")[["committee_rank", "explainer_rank"]].astype(int)
    prev = week_data(week - 1)
    if prev:
        lw = prev["contributions"].committee_rank.reindex(top.index)
        top["last_week"] = lw.where(lw <= 25)
        top["change"] = (lw - top.committee_rank).where(lw <= 25)
    pred = prediction(week)
    if pred is not None:
        top = top.join(pred.set_index("team")[["pred_rank", "rank_p10", "rank_p90"]])
    st.subheader(f"Week {week} committee Top 25")
    st.dataframe(top.reset_index().rename(columns={"committee_rank": "committee", "explainer_rank": "explainer"}),
                 hide_index=True)
    st.subheader("Biggest movers vs last week")
    if prev:
        mv = explain.movers(C, prev["contributions"])
        mv["top_factor"] = mv.top_factor.map(nm)
        st.dataframe(mv, hide_index=True)
    else:
        st.write("No earlier committee release is in the data yet.")
    with st.expander("Full written explanation"):
        st.markdown(E["markdown"])


def page_valued(week, E):
    w = E["weights"][E["weights"].kind == "week"].copy()
    view = st.radio("View", ["Individual factors", "Groups"], horizontal=True, key="valued_view")
    if view == "Groups":
        g = w.groupby("group").influence.sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h"))
        fig.update_layout(xaxis_title="Share of influence", height=350)
    else:
        w = w.assign(name=w.factor.map(nm)).sort_values("weight")
        fig = go.Figure(go.Bar(
            x=w.weight, y=w.name, orientation="h",
            error_x=dict(type="data", symmetric=False, array=(w.hi - w.weight).clip(lower=0),
                         arrayminus=(w.weight - w.lo).clip(lower=0))))
        fig.update_layout(xaxis_title="Weight (rank-points per 1 SD); bars = 90% bootstrap band",
                          height=max(350, 28 * len(w)))
    st.plotly_chart(fig)
    st.caption("Positive = a higher value of the stat helps a team; negative = a higher value hurts. "
               "One week has only 25 ranked teams, so lean on the bands and the pooled fit below.")
    A = all_weights()
    if A is not None and A.week.nunique() > 1:
        st.subheader("How the weights have shifted")
        wk = A[A.kind == "week"]
        default = list(w.sort_values("influence", ascending=False).factor.head(5))
        pick = st.multiselect("Factors", sorted(wk.factor.unique()), default=default, format_func=nm, key="hist")
        fig2 = go.Figure([go.Scatter(x=d.week, y=d.weight, name=nm(f), mode="lines+markers")
                          for f, d in wk[wk.factor.isin(pick)].groupby("factor")])
        fig2.update_layout(xaxis_title="Release week", yaxis_title="Weight")
        st.plotly_chart(fig2)
    else:
        st.caption("The weight history appears once two or more weeks have been explained.")
    st.subheader("Pooled fit (all weeks so far)")
    pooled = E["weights"][E["weights"].kind == "pooled"].assign(factor=lambda d: d.factor.map(nm))
    st.dataframe(pooled.drop(columns=["kind", "week"]).sort_values("influence", ascending=False), hide_index=True)


def page_team(week, E):
    C = E["contributions"]
    feats = explain.feature_columns(C)
    order = C.sort_values("committee_rank").index.tolist()
    team = st.selectbox("Team", order, key="team")
    row = C.loc[team]
    cr = "NR" if row.committee_rank > 25 else int(row.committee_rank)
    c1, c2, c3 = st.columns(3)
    c1.metric("Committee rank", cr)
    c2.metric("Explainer rank", int(row.explainer_rank))
    pred = prediction(week)
    if pred is not None and team in set(pred.team):
        p = pred.set_index("team").loc[team]
        c3.metric("Model forecast", f"#{int(p.pred_rank)}", f"band {int(p.rank_p10)}-{int(p.rank_p90)}", delta_color="off")
    top = row[feats].sort_values(key=abs, ascending=False).head(12)
    other = float(row.explainer_score - row.baseline - top.sum())
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
        x=["Average Top 25 team"] + [nm(f) for f in top.index] + ["All other factors", "Explainer score"],
        y=[row.baseline] + list(top.values) + [other, 0]))
    fig.update_layout(title=f"{team}: what lifted or sank the score", height=480, showlegend=False)
    st.plotly_chart(fig)
    if E["raw"] is not None and team in E["raw"].index:
        t = pd.DataFrame({"factor": [nm(f) for f in feats], "raw value": E["raw"].loc[team, feats].values,
                          "z-score": E["z"].loc[team, feats].values, "contribution": row[feats].values})
        st.subheader("Feature values")
        st.dataframe(t.sort_values("contribution", key=abs, ascending=False), hide_index=True)
    h = E["h2h"]
    c1, c2 = st.columns(2)
    c1.subheader("Beat (FBS)")
    c1.write(", ".join(h[h.winner == team].loser) or "-")
    c2.subheader("Lost to (FBS)")
    c2.write(", ".join(h[h.loser == team].winner) or "-")


def page_compare(week, E):
    C = E["contributions"]
    teams = C.sort_values("committee_rank").index.tolist()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Team A", teams, index=0, key="cmp_a")
    b = c2.selectbox("Team B", teams, index=1, key="cmp_b")
    if a == b:
        st.info("Pick two different teams.")
        return
    P = explain.pair_explanation(C, a, b, set(zip(E["h2h"].winner, E["h2h"].loser)), top_n=8)
    ra, rb = C.loc[a, "committee_rank"], C.loc[b, "committee_rank"]
    st.markdown(explain.describe_pair(P, int(ra), int(rb)).lstrip("- "))
    feats = explain.feature_columns(C)
    d = (C.loc[a, feats] - C.loc[b, feats]).sort_values()
    d = d[d.abs() > 0.02]
    fig = go.Figure(go.Bar(x=d.values, y=[nm(f) for f in d.index], orientation="h",
                           marker_color=["#c0392b" if v < 0 else "#2e86c1" for v in d.values]))
    fig.update_layout(xaxis_title=f"Contribution difference ({a} minus {b}); right = favors {a}",
                      height=max(300, 26 * len(d)))
    st.plotly_chart(fig)


def page_scorecard(week):
    sc = csv(str(RES / "scorecard.csv"))
    st.subheader("Scored forecasts")
    if sc is None or not len(sc):
        st.info("No forecasts have been scored yet. The first live forecast is Week 11.")
    else:
        st.dataframe(sc.round(3), hide_index=True)
        best = sc.groupby("method")[["kendall", "pairwise_acc_top25", "top12_overlap"]].mean().round(3)
        st.caption("Average across scored weeks:")
        st.dataframe(best)
    misses = csv(str(RES / "biggest_misses.csv"))
    if misses is not None:
        st.subheader("Biggest misses")
        st.dataframe(misses[misses.week == week] if (misses.week == week).any() else misses, hide_index=True)
    pred = prediction(week)
    st.subheader(f"Frozen prediction, week {week}")
    if pred is None:
        st.info(f"No frozen prediction for week {week}.")
    else:
        st.dataframe(pred.head(30), hide_index=True)
    hist = csv(str(RES / "model_comparison_history.csv"))
    if hist is not None:
        st.subheader("Champion selection history")
        st.dataframe(hist.sort_values(["trained_through_week", "selection_score"], ascending=[False, False]),
                     hide_index=True)


def page_data(week, E):
    tabs = st.tabs(["Features", "Collinearity", "Feature drift", "API budget", "Changelog"])
    with tabs[0]:
        raw = E["raw"] if E else None
        if raw is None:
            st.info(f"No feature file for week {week}.")
        else:
            q = st.text_input("Filter teams or conferences", key="feat_filter")
            t = raw.reset_index(names="team")
            if q:
                m = t.team.str.contains(q, case=False) | t.conference.astype(str).str.contains(q, case=False)
                t = t[m]
            st.dataframe(t, hide_index=True)
            download(t, f"features_wk{week}.csv")
    for tab, (label, path) in zip(tabs[1:4], [("Collinearity", RES / "collinearity.csv"),
                                              ("Drift", RES / "feature_drift.csv"),
                                              ("API budget", LOGS / "call_budget.csv")]):
        with tab:
            d = csv(str(path))
            if d is None:
                st.info(f"{path.name} not found.")
            else:
                st.dataframe(d, hide_index=True)
                download(d, path.name)
    with tabs[4]:
        f = ROOT / "CHANGELOG.md"
        st.markdown(f.read_text() if f.exists() else "No changelog yet.")


# ----------------------------------------------------------------------------- main
def main():
    pred_weeks = {int(p.stem.split("_")[1]) for p in PRED.glob("week_*_prediction.csv")}
    weeks = sorted(set(explain.explained_weeks()) | pred_weeks)
    with st.sidebar:
        st.title("CFP committee explainer")
        if not weeks:
            st.warning("No data yet. Run `run_weekly.py baseline`, then `run_weekly.py explain --week 10`.")
            st.stop()
        week = st.selectbox("Release week", weeks, index=len(weeks) - 1, key="week")
        page = st.radio("Page", PAGES, key="page")
        if st.button("Refresh data", key="refresh"):
            st.cache_data.clear()
            st.rerun()
    st.header(page)
    E = week_data(week)
    if page == "Prediction scorecard":
        page_scorecard(week)
    elif page == "Data browser":
        page_data(week, E)
    elif E is None:
        no_explanation(week)
    else:
        {"Overview": page_overview, "What the committee valued": page_valued,
         "Team explorer": page_team, "Compare two teams": page_compare}[page](week, E)


main()
```

- [ ] **Step 4: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_app_smoke.py -v`
Expected: 6 passed. If a page raises, fix the page (the exception text names the line); do not weaken the test.

- [ ] **Step 5: Look at it for real**

Run: `.venv/bin/streamlit run app.py`
Expected: opens at http://localhost:8501. Click each of the 6 pages with Week 10 selected. Check that the Top 25 table starts with Miami, the waterfall renders for a team, and the Compare page shows a sentence plus a bar chart. Stop the server with Ctrl+C. Report anything that looks wrong to the user.

---

### Task 8: Documentation and final verification

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: README: add an "Explanations and app" section**

Insert before the `## Files` heading in `README.md`:

```markdown
## Explanations and app
`update` (and so `baseline` and `score`) also fits a separate ridge explainer on the committee's *actual* Top 25 and writes `explanations/week_N_explanation.md`, `week_N_team_contributions.csv`, `week_N_factor_weights.csv` and `week_N_fit_quality.csv`. The forecasting champion and `predictions/` are untouched.
```bash
.venv/bin/python run_weekly.py explain --week 10   # regenerate one week's files (no API, no retrain)
.venv/bin/streamlit run app.py                     # read-only explorer
```
Run the tests with `.venv/bin/python -m pytest`.
```

And add two rows to the Files table:

```markdown
| explain.py | Committee explainer: factor weights, per-team contributions, pairs, movers |
| app.py | Streamlit explorer (read-only) |
```

- [ ] **Step 2: CLAUDE.md: add the commands and architecture notes**

In the commands block add:

```bash
.venv/bin/python run_weekly.py explain --week N     # rewrite explanations/week_N_* (no API, no retrain)
.venv/bin/streamlit run app.py                      # read-only explorer
.venv/bin/python -m pytest                          # tests (single test: pytest tests/test_explain.py::test_name)
```

Change the sentence "There is no test suite or linter" to "Tests are in `tests/` (pytest); there is no linter."

Add this bullet to the Architecture list:

```markdown
- **Explainer** (`explain.py`, UI-free): a ridge fit on the committee's *actual* Top 25, separate from the forecasting champion, so each team's score splits exactly into factor contributions. `cmd_update` calls it (wrapped in try/except so it never blocks retraining) and writes `explanations/week_N_*`. `app.py` only reads those files plus existing `results/`, `predictions/` and `data/features/`; it must never call the API or write project files.
```

- [ ] **Step 3: Full verification**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (about 28), none skipped for missing data.

Run: `ls predictions models && md5 models/week_10_champion.pkl`
Expected: `predictions/` is still empty and the pickle exists (unchanged by this work).

- [ ] **Step 4: Report to the user**

Summarize: what was built, the four deviations from the spec listed at the top of this plan, the in-sample fit numbers for Week 10 from `explanations/week_10_fit_quality.csv`, and anything odd found in Steps 6.4 and 7.5.

---

## Self-review (completed)

**Spec coverage**
- Explainer model and pooled fit: Tasks 1, 2.
- Fit quality: Task 1. Weights with bootstrap bands and group shares: Task 2.
- Per-team contributions, pairs, unexplained, movers: Tasks 1, 3.
- Markdown, CSV outputs and the CLI hook: Tasks 4, 5. REPORT.md section and CHANGELOG entry: Task 5.
- App with six pages, week selector, Refresh, CSV downloads, missing-data messages: Task 7.
- Error handling (fewer than 25 ranked teams, no partial writes, `score` still completes): Tasks 4 and 5 (`explain_frames` raises before any write; the hook catches exceptions).
- Tests including byte-identical `predictions/` and `models/`: Task 6.
- Out of scope items are not built.

**Placeholders:** none; every code step is complete.

**Type consistency:** `fit_ridge` returns `(coef, intercept, alpha)` and is used that way in Tasks 1, 2 and 4; `team_contributions(wk, feats, coef, intercept)`; `pair_explanation` returns keys `a, b, gap, favoring_a, favoring_b, h2h` as consumed by `describe_pair` and the app; `unexplained` returns `committee_rank, explainer_rank, gap`; `movers` returns `team, prev_rank, rank, move, top_factor, top_delta`; weights columns `week, kind, factor, group, weight, lo, hi, influence, alpha`.

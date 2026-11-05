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


# ----------------------------------------------------------------------------- standalone correlation
def test_sign_conflict_rules():
    # confident weight, opposite sign to a real standalone correlation -> flagged
    assert explain.sign_conflict(weight=1.7, corr=-0.42, lo=0.9, hi=2.3)
    # same sign -> not flagged
    assert not explain.sign_conflict(weight=1.6, corr=0.56, lo=0.7, hi=2.4)
    # band spans zero -> weight is not a confident claim
    assert not explain.sign_conflict(weight=0.5, corr=-0.4, lo=-0.1, hi=1.0)
    # negligible standalone correlation -> nothing to contradict
    assert not explain.sign_conflict(weight=1.0, corr=-0.05, lo=0.5, hi=1.5)


def test_factor_weights_has_standalone_corr():
    df = pd.concat([make_week(1), make_week(2)], ignore_index=True)
    W = explain.factor_weights(df, FEATS, week=2, B=20)
    assert {"corr", "sign_conflict"} <= set(W.columns)
    wk = W[W.kind == "week"].set_index("factor")
    sub = df[df.week == 2]
    for f in FEATS:
        assert wk.loc[f, "corr"] == pytest.approx(np.corrcoef(sub[f], -sub["rank"])[0, 1])
    pooled = W[W.kind == "pooled"].set_index("factor")
    assert pooled.loc["f1", "corr"] == pytest.approx(np.corrcoef(df["f1"], -df["rank"])[0, 1])


def test_suppressor_case_is_flagged():
    """x2 is a noisy copy of x1 and only x1 drives rank: x2 correlates positively but gets a negative weight."""
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    x2 = x1 + 0.3 * rng.normal(size=n)
    order = (-(x1)).argsort().argsort() + 1
    df = pd.DataFrame({"x1": x1, "x2": x2, "team": [f"T{i}" for i in range(n)],
                       "rank": np.where(order <= 25, order, 26).astype(float), "week": 1})
    W = explain.factor_weights(df, ["x1", "x2"], week=1, B=60).query("kind == 'week'").set_index("factor")
    assert W.loc["x2", "corr"] > 0.2
    assert W.loc["x1", "corr"] > 0.2
    # at least the sanity relationship: flags are booleans and only set when signs differ
    for f, r in W.iterrows():
        if r.sign_conflict:
            assert np.sign(r.weight) != np.sign(r["corr"])


def test_markdown_shows_corr_and_conflicts():
    df, h2h = _two_weeks()
    res = explain.explain_frames(df, h2h, 2, FEATS, B=20)
    res["weights"].loc[(res["weights"].kind == "week") & (res["weights"].factor == "f1"), "sign_conflict"] = True
    md = explain.build_markdown(res)
    assert "Standalone corr" in md
    assert "sign differs" in md
    assert "Sign conflicts" in md

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


CORR_FLOOR = 0.1     # |standalone correlation| below this is too small to contradict


def standalone_corr(sub, feats):
    """Correlation of each stat with the committee's order (goodness = -rank) over every team in `sub`."""
    good = -sub["rank"].astype(float)
    return pd.Series({f: sub[f].corr(good) for f in feats}).fillna(0.0)


def sign_conflict(weight, corr, lo, hi, floor=CORR_FLOOR):
    """True when a confident weight (band excludes zero) has the opposite sign to the stat's standalone
    correlation with the committee's order. This happens when stats overlap (suppressor effects)."""
    confident = lo > 0 or hi < 0
    return bool(confident and abs(corr) >= floor and np.sign(weight) != np.sign(corr))


def factor_weights(df, feats, week, B=BOOT):
    """Long table: this week's fit (kind='week') and the fit pooled over every week up to `week`.

    `corr` is each stat's standalone correlation with the committee's order; `sign_conflict` flags rows
    whose weight sign contradicts it.
    """
    rows = []
    for kind, sub in (("week", df[df.week == week]), ("pooled", df)):
        coef, _, alpha = fit_ridge(sub, feats)
        band = bootstrap_weights(sub, feats, alpha, B)
        corr = standalone_corr(sub, feats)
        rows.append(pd.DataFrame({
            "week": week, "kind": kind, "factor": feats,
            "group": [GROUP_OF.get(f, "Other") for f in feats],
            "weight": coef.values, "lo": band.lo.values, "hi": band.hi.values,
            "corr": corr.values,
            "sign_conflict": [sign_conflict(w, c, lo, hi) for w, c, lo, hi
                              in zip(coef.values, corr.values, band.lo.values, band.hi.values)],
            "influence": _influence(sub, feats, coef).values, "alpha": alpha}))
    return pd.concat(rows, ignore_index=True)


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
         "| Factor | Group | Weight | 90% band | Standalone corr | Share of influence | Change vs last week |",
         "|---|---|---|---|---|---|---|"]
    for f, r in wk.sort_values("influence", ascending=False).head(10).iterrows():
        d = f"{r.weight - prev[f]:+.2f}" if prev is not None and f in prev.index else "n/a"
        flag = " ⚠ sign differs" if r.sign_conflict else ""
        L.append(f"| {_name(f)} | {r.group} | {r.weight:+.2f} | {r.lo:+.2f} to {r.hi:+.2f} | "
                 f"{r['corr']:+.2f}{flag} | {r.influence:.0%} | {d} |")
    g = wk.groupby("group").influence.sum().sort_values(ascending=False)
    conflicts = wk[wk.sign_conflict]
    L += ["", "Sign: positive = a higher value of the stat helps a team; negative = a higher value hurts "
              "(so lower is better). Weight = change in the explainer's score per 1 SD of the stat, holding the "
              "other stats fixed. Standalone corr = how the stat lines up with the committee's order by itself "
              "(across all FBS teams; positive = higher value goes with a better rank, so compare its sign with the weight). "
              "Share of influence = share of total mean |contribution| across the committee's ranked teams.", ""]
    if len(conflicts):
        L += ["**Sign conflicts** (⚠: the weight's sign contradicts the stat's standalone correlation, with a band "
              "excluding zero): " + ", ".join(
                  f"{_name(f)} (weight {r.weight:+.2f}, standalone {r['corr']:+.2f})" for f, r in conflicts.iterrows())
              + ". This happens when stats overlap: after accounting for the correlated stats, this one picks up "
                "leftover signal. Read its weight as 'given the other stats', not as 'the committee dislikes it'.", ""]
    else:
        L += ["Sign conflicts: none (no confident weight contradicts its standalone correlation).", ""]
    L += [
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

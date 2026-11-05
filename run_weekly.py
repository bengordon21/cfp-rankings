"""Weekly cycle: predict -> score -> update.

  python run_weekly.py baseline            # Week 10 baseline (Steps 1-3), no scoring
  python run_weekly.py predict --week 11   # Sun/Mon: frozen pre-show prediction
  python run_weekly.py score   --week 11   # Tue night: score vs committee, then update
  python run_weekly.py update  --week 11   # retrain / pick champion (score runs this)
  python run_weekly.py explain --week 11   # rewrite explanations/week_11_* (no retrain, no API)
  python run_weekly.py log --message "..." --reason "..."   # manual tweak -> CHANGELOG.md
"""
import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_features as bf
import explain as explainer      # aliased: this module defines its own explain() helper
import fetch
import models as M
import report

ROOT = Path(__file__).parent
PRED = ROOT / "predictions"
RES = ROOT / "results"
MOD = ROOT / "models"
TRAIN = ROOT / "data" / "training"
FIRST_WEEK = 10
BOOT = 200


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- bookkeeping
def net_calls():
    if not fetch.LOG.exists():
        return 0
    return int((pd.read_csv(fetch.LOG).network_call == 1).sum())


def planned_release_calls(n):
    names = ["teams_fbs", "conferences", f"games_rel{n}", f"stats_adv_rel{n}"] + [f"ratings_{r}_rel{n}" for r in ("sp", "srs", "elo", "fpi")]
    return sum(not (fetch.RAW / f"{x}.json").exists() for x in names)


def log_budget(command, week, planned, before):
    actual = net_calls() - before
    f = ROOT / "logs" / "call_budget.csv"
    new = not f.exists()
    rem = fetch.info().get("remainingCalls")
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc_time", "command", "week", "planned_calls", "actual_calls", "remaining_monthly_calls"])
        w.writerow([now(), command, week, planned, actual, rem])
    print(f"API calls: planned {planned}, actual {actual}, remaining this month {rem}")


def changelog(title, body, reason):
    f = ROOT / "CHANGELOG.md"
    if not f.exists():
        f.write_text("# Changelog\n\nEvery model/feature/process change, with the reason. "
                     "Manual tweaks are tagged **[manual]**.\n")
    with open(f, "a") as fh:
        fh.write(f"\n## {now()} - {title}\n{body}\n**Reason:** {reason}\n")


# ----------------------------------------------------------------------------- actuals
def load_actuals():
    f = TRAIN / "actuals.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["week", "team", "rank"])


def save_actuals(week, ranks):
    TRAIN.mkdir(parents=True, exist_ok=True)
    a = load_actuals()
    a = a[a.week != week]
    new = pd.DataFrame({"week": week, "team": list(ranks), "rank": list(ranks.values())})
    pd.concat([a, new]).sort_values(["week", "rank"]).to_csv(TRAIN / "actuals.csv", index=False)


def get_actual_ranks(week, csv_path=None):
    if csv_path:
        d = pd.read_csv(csv_path)
        d.columns = [c.lower() for c in d.columns]
        team_col = "team" if "team" in d else "school"
        return dict(zip(d[team_col], d["rank"].astype(int)))
    fetch.fetch_committee(week)
    r = bf.committee_ranks(week)
    if not r:
        sys.exit(f"Committee poll for week {week} is not available from the API yet. "
                 "Provide the Top 25 as a CSV (columns: rank,team) via --csv.")
    return r


# ----------------------------------------------------------------------------- update
def latest_champion_path(max_week):
    c = sorted(MOD.glob("week_*_champion.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    c = [p for p in c if int(p.stem.split("_")[1]) <= max_week]
    return c[-1] if c else None


def cmd_update(week):
    MOD.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    fetch.fetch_release(week)                      # no-op when cached
    actuals = load_actuals()
    actuals = actuals[actuals.week <= week]
    if week not in set(actuals.week):
        sys.exit(f"No committee ranking stored for week {week}; run score (or baseline) first.")
    try:                                            # explanation never blocks retraining
        _, paths = explainer.run(week, actuals)
        changelog(f"Week {week} explanation written",
                  f"- {', '.join(p.name for p in paths.values())} in explanations/.",
                  "Explainer refit on the committee's actual Top 25 (explain.py); the forecasting champion is unchanged.")
        print(f"Explanation: {paths['md']}")
    except Exception as e:
        print(f"Explanation skipped: {e}")
    prev_path = latest_champion_path(week - 1)
    prev = joblib.load(prev_path) if prev_path else None
    df, h2h = M.load_training(actuals)
    weeks = [int(w) for w in sorted(df.week.unique())]
    print(f"Evaluating candidates on weeks {weeks} ...")
    per_week, kept, vif_log = M.evaluate_candidates(df, h2h, log=lambda *a: None)
    S = M.summarize(per_week, weeks)
    best = M.choose_champion(S)
    champ = M.fit_champion({k: best[k] for k in ("framing", "family", "param")}, kept, df, h2h, week,
                           cv={k: best[k] for k in ("kendall", "spearman", "pairwise_acc", "pairwise_acc_top25",
                                                    "top25_overlap", "top12_overlap", "selection_score", "scheme")})
    joblib.dump(champ, MOD / f"week_{week}_champion.pkl")

    # results tables ---------------------------------------------------------------
    S.insert(0, "trained_through_week", week)
    S.to_csv(RES / "model_comparison.csv", index=False)
    hist = RES / "model_comparison_history.csv"
    pd.concat([pd.read_csv(hist), S]) .to_csv(hist, index=False) if hist.exists() else S.to_csv(hist, index=False)
    per_week.to_csv(RES / "model_cv_by_week.csv", index=False)
    vif_log.assign(trained_through_week=week).to_csv(RES / "collinearity.csv", index=False)
    pd.DataFrame([(fr, f) for fr, fs in kept.items() for f in fs], columns=["framing", "kept_feature"]
                 ).assign(trained_through_week=week).to_csv(RES / "features_kept.csv", index=False)
    df[kept["combined"]].corr().round(3).to_csv(RES / "feature_correlations.csv")
    write_importance(champ, kept, df, h2h, S, week)
    write_drift(champ, df, h2h, week)

    old = f"{prev.version} (through wk{prev.trained_through_week})" if prev else "none"
    changelog(f"Week {week} update: champion {champ.version}",
              f"- Trained on weeks {weeks}; validation = {best['scheme']}. Previous champion: {old}.\n"
              f"- Champion selection score {best['selection_score']:.3f} "
              f"(Kendall {best['kendall']:.3f}, within-Top-25 pairwise acc {best['pairwise_acc_top25']:.3f}).\n"
              f"- Features kept ({len(champ.feats)}): {', '.join(champ.feats)}.\n"
              f"- VIF-dropped in {champ.spec['framing']}: "
              f"{', '.join(vif_log[vif_log.framing == champ.spec['framing']].dropped_feature) or 'none'}.",
              "Automatic retrain; champion = best mean leave-one-"
              f"{'week' if len(weeks) > 1 else 'team'}-out score over all past weeks (pairwise preferred within 0.01).")
    report.write_report()
    print(f"Champion: {champ.version}  (selection score {best['selection_score']:.3f}, {best['scheme']})")
    return champ


def write_importance(champ, kept, df, h2h, S, week):
    rows = []
    for fr in ("resume", "power", "combined"):
        for fam, grid in (("pairwise", None), ("gbm", None)):
            sub = S[(S.framing == fr) & (S.family == fam)]
            if not len(sub):
                continue
            b = sub.iloc[0]
            p = None if (fam == "gbm") else float(b.param)
            m = M.Model(fam, p, kept[fr]).fit(df, h2h); m._train = df
            imp = m.importance()
            for f, v in imp.items():
                rows.append({"trained_through_week": week, "framing": fr, "model": fam, "param": p,
                             "feature": f, "value": v,
                             "kind": "standardized_logit_coef" if fam == "pairwise" else "mean_abs_shap"})
    pd.DataFrame(rows).to_csv(RES / "feature_importance.csv", index=False)
    imp = champ.model.importance().sort_values()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.5))
    ax.barh(imp.index, imp.values, color=["#c0392b" if v < 0 else "#2e6da4" for v in imp.values])
    ax.set_title(f"{champ.version}: drivers of committee rank\n(standardized; right = ranked higher)")
    plt.tight_layout(); fig.savefig(RES / f"feature_importance_week{week}.png", dpi=130); plt.close(fig)
    if champ.spec["family"] != "gbm":
        g = M.Model("gbm", None, kept[champ.spec["framing"]]).fit(champ.train_df, champ.h2h)
        s = pd.Series(np.abs(g.contributions(champ.train_df)).mean().values, index=g.feats).sort_values()
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.5))
        ax.barh(s.index, s.values, color="#6c8e3f"); ax.set_title("Gradient boosting: mean |SHAP| (ranks)")
        plt.tight_layout(); fig.savefig(RES / f"shap_importance_week{week}.png", dpi=130); plt.close(fig)


def write_drift(champ, df, h2h, week):
    """Per-week refits of the champion's feature set: pairwise coefficients and GBM SHAP."""
    rows = []
    C = champ.spec["param"] if champ.spec["family"] == "pairwise" else 0.01
    for w in sorted(df.week.unique()):
        d = df[df.week == w]
        feats = [f for f in champ.feats if d[f].std() > 1e-9]
        pm = M.Model("pairwise", C, feats).fit(d, h2h)
        gm = M.Model("gbm", None, feats).fit(d, h2h)
        shap_abs = np.abs(gm.contributions(d)).mean()
        for f in feats:
            rows.append({"week": w, "model": "pairwise_coef", "feature": f, "value": float(pm.coef[feats.index(f)])})
            rows.append({"week": w, "model": "gbm_mean_abs_shap", "feature": f, "value": float(shap_abs[f])})
    D = pd.DataFrame(rows); D["champion_version"] = champ.version
    D.to_csv(RES / "feature_drift.csv", index=False)
    if D.week.nunique() >= 2:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mdl in zip(axs, ("pairwise_coef", "gbm_mean_abs_shap")):
            P = D[D.model == mdl].pivot(index="week", columns="feature", values="value")
            keep = P.abs().max().sort_values(ascending=False).head(8).index
            P[keep].plot(ax=ax, marker="o"); ax.set_title(mdl); ax.legend(fontsize=7)
        plt.tight_layout(); fig.savefig(RES / "feature_drift.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- predict
def conf_champions(X, score_rank):
    """Projected champion per conference = best conference record, tie -> better predicted rank."""
    out = {}
    for c, g in X.groupby("conference"):
        if c == "FBS Independents":
            continue
        g = g.assign(pr=score_rank.reindex(g.index))
        g = g.sort_values(["conf_win_pct", "conf_wins", "pr"], ascending=[False, False, True])
        out[c] = g.index[0]
    return out


def build_bracket(pred, X):
    ch = conf_champions(X, pred.set_index("team").pred_rank)
    champs = sorted(ch.values(), key=lambda t: pred.set_index("team").pred_rank[t])
    auto = champs[:5]
    rest = [t for t in pred.sort_values("pred_rank").team if t not in auto]
    at_large = rest[:7]
    field = sorted(auto + at_large, key=lambda t: pred.set_index("team").pred_rank[t])
    return ch, auto, at_large, field


def storylines(pred, R, X, auto, field):
    names = list(pred.sort_values("team").team)       # R columns follow pred order
    lines = []
    idx = {t: i for i, t in enumerate(pred.team)}
    top = pred.sort_values("pred_rank").head(20)
    for (_, a), (_, b) in zip(top.iloc[:-1].iterrows(), top.iloc[1:].iterrows()):
        p = float((R[:, idx[a.team]] < R[:, idx[b.team]]).mean())
        if 0.35 <= p <= 0.65:
            lines.append(f"- **Toss-up:** #{int(a.pred_rank)} {a.team} vs #{int(b.pred_rank)} {b.team} "
                         f"(model gives {a.team} only {p:.0%} to stay ahead).")
    pr = pred[pred.prev_committee_rank.notna()].copy()
    if len(pr):
        pr["move"] = pr.prev_committee_rank - pr.pred_rank
        for _, r in pr.reindex(pr.move.abs().sort_values(ascending=False).index).head(5).iterrows():
            if abs(r.move) >= 3:
                lines.append(f"- **Mover:** {r.team} {int(r.prev_committee_rank)} -> predicted {int(r.pred_rank)} "
                             f"({'up' if r.move > 0 else 'down'} {int(abs(r.move))}); {X.loc[r.team, 'wins']:.0f}-{X.loc[r.team, 'losses']:.0f}.")
    bub = pred[(pred.p_top12 > 0.15) & (pred.p_top12 < 0.85)].sort_values("pred_rank")
    if len(bub):
        lines.append("- **Bubble (12-team field):** " + ", ".join(
            f"{r.team} (#{int(r.pred_rank)}, {r.p_top12:.0%} in top 12)" for _, r in bub.head(8).iterrows()))
    fifth = [t for t in auto][-1:]
    if fifth:
        r = pred.set_index("team").loc[fifth[0]]
        lines.append(f"- **Last auto-bid:** projected 5th champion is {fifth[0]} (predicted #{int(r.pred_rank)}).")
    return lines or ["- (no clear storylines flagged)"]


def cmd_predict(week, redo=False):
    PRED.mkdir(exist_ok=True)
    out_csv, out_md = PRED / f"week_{week}_prediction.csv", PRED / f"week_{week}_prediction.md"
    sc = RES / "scorecard.csv"
    scored = sc.exists() and (pd.read_csv(sc).week == week).any()
    if scored:
        sys.exit(f"Week {week} has already been scored; its prediction is frozen and cannot be rewritten.")
    if out_csv.exists():
        if not redo:
            sys.exit(f"{out_csv.name} already exists (frozen). Use --redo only before the show; "
                     "the old file is archived.")
        (PRED / "archive").mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for f in (out_csv, out_md):
            if f.exists():
                shutil.move(f, PRED / "archive" / f"{f.stem}_{stamp}{f.suffix}")
    cp = latest_champion_path(week - 1)
    if cp is None:
        sys.exit("No champion model found. Run `python run_weekly.py baseline` first.")
    champ = joblib.load(cp)
    plan, before = planned_release_calls(week), net_calls()
    fetch.fetch_release(week)
    log_budget("predict", week, plan, before)

    X, hh = bf.get_features(week)
    if X.prev_committee_rank.notna().any():
        X["prev_committee_rank"] = X.prev_committee_rank.fillna(M.UNRANKED)
    Z = bf.zscore(X); Z["team"] = Z.index; Z = Z.reset_index(drop=True)
    for f in champ.feats:
        if f not in Z:
            Z[f] = 0.0
    h2h_w = set(zip(hh.winner, hh.loser))
    m = champ.model
    s = m.score(Z, h2h_w)
    R = M.bootstrap_ranks(champ, Z, h2h_w, B=BOOT)
    contrib = m.contributions(Z)

    pred = pd.DataFrame({"team": Z.team, "conference": X.conference.values, "score": s})
    pred["pred_rank"] = (-pred.score).rank(method="first").astype(int)
    pred["rank_p10"] = np.percentile(R, 10, axis=0); pred["rank_p50"] = np.percentile(R, 50, axis=0)
    pred["rank_p90"] = np.percentile(R, 90, axis=0)
    pred["p_top25"] = (R <= 25).mean(0); pred["p_top12"] = (R <= 12).mean(0)
    pred["record"] = [f"{int(X.loc[t, 'wins'])}-{int(X.loc[t, 'losses'])}" for t in pred.team]
    pred["sp"] = X.sp.reindex(pred.team).values
    pred["prev_committee_rank"] = X.prev_committee_rank.reindex(pred.team).values
    pred["weeks_since_loss"] = X.weeks_since_loss.reindex(pred.team).values

    def drivers(t):
        c = contrib.loc[t].sort_values(ascending=False)
        up = [f"{k} {v:+.2f}" for k, v in c.head(3).items() if v > 0.02]
        dn = [f"{k} {v:+.2f}" for k, v in c.tail(2).items() if v < -0.02]
        return "; ".join(up) + (" | " if up and dn else "") + "; ".join(dn)
    pred["drivers"] = [drivers(t) for t in pred.team]
    pred = pred.sort_values("pred_rank").reset_index(drop=True)
    pred["model_version"] = champ.version
    pred["generated_utc"] = now()
    pred["data_through_cfbd_week"] = week - 1

    R = R[:, [list(Z.team).index(t) for t in pred.team]]       # align boot columns to pred order
    ch, auto, at_large, field = build_bracket(pred, X)
    pred["projected_conf_champ"] = pred.team.isin(ch.values())
    pred["in_projected_field"] = pred.team.isin(field)
    pred.round(4).to_csv(out_csv, index=False)

    md = [f"# Week {week} committee prediction (FROZEN)",
          f"Generated {pred.generated_utc[0]} UTC | model `{champ.version}` | data through CFBD week {week - 1} "
          f"(games completed before the release) | {BOOT} bootstrap refits for bands.",
          "", "## Predicted Top 25", "",
          "| Rank | Team | Rec | Band (p10-p90) | P(top 12) | Main drivers (z-score contributions) |",
          "|---|---|---|---|---|---|"]
    for _, r in pred.head(25).iterrows():
        md.append(f"| {r.pred_rank} | {r.team} | {r.record} | {int(r.rank_p10)}-{int(r.rank_p90)} | "
                  f"{r.p_top12:.0%} | {r.drivers} |")
    md += ["", "Next 5 out: " + ", ".join(f"{r.team} ({r.pred_rank})" for _, r in pred.iloc[25:30].iterrows()),
           "", "## Projected 12-team bracket",
           "_Auto bids: 5 highest-ranked projected conference champions (leader by conference record; "
           "Independents excluded). Then 7 highest-ranked others. Seeded 1-12 by predicted rank (verify the "
           "2026 seeding/bye rules - assumed straight seeding)._", ""]
    for seed, t in enumerate(field, 1):
        r = pred.set_index("team").loc[t]
        tag = "auto (champion)" if t in auto else "at-large"
        md.append(f"{seed}. {t} ({r.record}, pred #{int(r.pred_rank)}) - {tag}")
    md += ["", "First round: 5 v 12, 6 v 11, 7 v 10, 8 v 9. Byes: seeds 1-4.", "",
           "## Storylines to watch", ""] + storylines(pred, R, X, auto, field)
    md += ["", "## Caveats", "",
           f"- Early-season model trained on {len(champ.train_df.week.unique())} committee release(s); "
           "bands are bootstrap-based and understate true uncertainty.",
           "- Ratings are snapshots taken when this was run; conference champions are projections, not finals."]
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md[:32]))
    print(f"\nSaved {out_csv.name} and {out_md.name}")


# ----------------------------------------------------------------------------- score
def explain(t, row, pred, actual, X, hh):
    """Plausible, data-based hypotheses for a big miss (not proof)."""
    p, a = int(row.pred_rank), int(row.actual_rank)
    better = a < p                                           # committee higher than model
    out = []
    pr = pred.set_index("team")
    wins = {l for (w, l) in hh if w == t}
    losses = {w for (w, l) in hh if l == t}
    if better:
        for o in sorted(wins, key=lambda o: pr.pred_rank.get(o, 999)):
            if pr.pred_rank.get(o, 999) < p and actual.get(o, 26) > a:
                out.append(f"committee honored the head-to-head win over {o} (model had {o} #{int(pr.pred_rank[o])})")
                break
    else:
        for o in sorted(losses, key=lambda o: actual.get(o, 26)):
            if pr.pred_rank.get(o, 999) > p and actual.get(o, 26) < a:
                out.append(f"committee weighed the head-to-head loss to {o} (model had {o} #{int(pr.pred_rank[o])})")
                break
    wsl = X.weeks_since_loss.get(t, 9)
    if wsl <= 1 and not better:
        out.append("punished a loss from the last week or two more than the model expected")
    if wsl <= 1 and better:
        out.append("treated a recent loss more leniently than the model")
    pv = row.prev_committee_rank
    if pd.notna(pv) and abs(a - pv) + 2 <= abs(p - pv):
        out.append(f"committee inertia: moved only {int(pv)} -> {a} while the model moved to {p}")
    if X.bad_losses.get(t, 0) > 0 and not better:
        out.append("a 'bad loss' (to a weak/unranked team) may have counted more than modeled")
    if X.losses.get(t, 0) == 0 and better:
        out.append("unbeaten record likely valued above what ratings imply")
    if X.is_p4.get(t, 1) == 0 and not better:
        out.append("possible Group-of-5 discount / schedule-strength skepticism")
    if X.is_p4.get(t, 1) == 0 and better:
        out.append("received more credit than modeled for a strong record outside the Power 4")
    if X.quality_wins_sp25.get(t, 0) >= 2 and better:
        out.append(f"{int(X.quality_wins_sp25[t])} wins over SP+ top-25 teams may have mattered more than modeled")
    return "; ".join(out[:3]) or "no clear data-based explanation (possible subjective/eye-test factors)"


def cmd_score(week, csv_path=None, no_update=False):
    out_csv = PRED / f"week_{week}_prediction.csv"
    if not out_csv.exists():
        sys.exit(f"No frozen prediction for week {week}; cannot score honestly.")
    pred = pd.read_csv(out_csv)
    plan, before = (1 if not (fetch.RAW / f"cfp_rankings_wk{week}.json").exists() and not csv_path else 0), net_calls()
    actual = get_actual_ranks(week, csv_path)
    if csv_path:
        # also store as the same cache format used by build_features.committee_ranks
        import json
        (fetch.RAW / f"cfp_rankings_wk{week}.json").write_text(json.dumps(
            [{"season": 2026, "seasonType": "regular", "week": week, "polls": [{
                "poll": "Playoff Committee Rankings", "isFinal": False,
                "ranks": [{"rank": r, "school": t} for t, r in actual.items()]}]}]))
    ap = None
    try:
        plan += 0 if (fetch.RAW / f"polls_all_wk{week}.json").exists() else 1
        polls = fetch.fetch_polls(week)
        for w in polls:
            if w["week"] == week:
                for p in w["polls"]:
                    if p["poll"] == "AP Top 25":
                        ap = {r["school"]: r["rank"] for r in p["ranks"]}
    except Exception as e:
        print("AP poll comparison unavailable:", e)
    log_budget("score", week, plan, before)

    teams = pred.team.tolist()
    rank = np.array([actual.get(t, M.UNRANKED) for t in teams], float)
    missing = [t for t in actual if t not in set(teams)]
    if missing:
        print("WARNING committee teams not found among FBS teams:", missing)
    X, hh = bf.get_features(week)
    sp = X.sp.reindex(teams).values
    prev = pred.prev_committee_rank.fillna(M.UNRANKED).values
    methods = {"model": pred.score.values, "baseline_sp_order": sp}
    if pred.prev_committee_rank.notna().any():
        methods["baseline_last_week_committee"] = -prev + 1e-6 * sp     # SP+ only breaks ties among unranked
    if ap:
        methods["comparison_AP_top25_order"] = np.array([-(ap.get(t, 26)) for t in teams], float) + 1e-6 * sp
    rows = []
    for k, sc_ in methods.items():
        rows.append({"week": week, "method": k, "model_version": pred.model_version[0],
                     "scored_utc": now(), **M.week_metrics(rank, sc_)})
    sc = pd.DataFrame(rows)
    f = RES / "scorecard.csv"
    if f.exists():
        old = pd.read_csv(f); old = old[old.week != week]; sc = pd.concat([old, sc])
    sc.to_csv(f, index=False)

    pr = pred.assign(actual_rank=rank)
    err = pr.assign(error=pr.pred_rank - pr.actual_rank)
    err = err[(err.pred_rank <= 25) | (err.actual_rank <= 25)]
    err["abs_err"] = err.error.abs()
    big = err.sort_values("abs_err", ascending=False).head(10)
    big = big.assign(week=week, explanation=[explain(r.team, r, pred, actual, X, set(zip(hh.winner, hh.loser)))
                                             for _, r in big.iterrows()])
    cols = ["week", "team", "pred_rank", "actual_rank", "error", "prev_committee_rank", "explanation"]
    bf_ = RES / "biggest_misses.csv"
    if bf_.exists():
        o = pd.read_csv(bf_); o = o[o.week != week]; big_all = pd.concat([o, big[cols]])
    else:
        big_all = big[cols]
    big_all.to_csv(bf_, index=False)

    md = [f"# Week {week} score", "", sc[sc.week == week].drop(columns=["model_version", "scored_utc", "week"]).round(3).to_markdown(index=False),
          "", "## Biggest misses (positive error = model had them lower than the committee)", ""]
    for _, r in big.iterrows():
        md.append(f"- **{r.team}**: predicted {int(r.pred_rank)}, committee {int(r.actual_rank)} "
                  f"({int(r.error):+d}). Plausible: {r.explanation}.")
    (RES / f"week_{week}_score.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    save_actuals(week, actual)
    if not no_update:
        cmd_update(week)


def cmd_explain(week):
    """Regenerate explanations/week_N_* from stored actuals. No API calls, no retraining."""
    _, paths = explainer.run(week, load_actuals())
    print("Wrote " + ", ".join(p.name for p in paths.values()))


def cmd_baseline():
    plan, before = planned_release_calls(FIRST_WEEK), net_calls()
    fetch.fetch_release(FIRST_WEEK)
    ranks = get_actual_ranks(FIRST_WEEK)
    log_budget("baseline", FIRST_WEEK, plan, before)
    save_actuals(FIRST_WEEK, ranks)
    cmd_update(FIRST_WEEK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "predict", "score", "update", "explain", "log"])
    ap.add_argument("--week", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--message"); ap.add_argument("--reason")
    a = ap.parse_args()
    if a.cmd == "baseline":
        cmd_baseline()
    elif a.cmd == "log":
        if not (a.message and a.reason):
            sys.exit("--message and --reason are both required")
        changelog("[manual] " + a.message, "", a.reason)
    else:
        if not a.week:
            sys.exit("--week N is required")
        {"predict": lambda: cmd_predict(a.week, a.redo),
         "score": lambda: cmd_score(a.week, a.csv, a.no_update),
         "update": lambda: cmd_update(a.week),
         "explain": lambda: cmd_explain(a.week)}[a.cmd]()

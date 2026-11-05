"""Generate REPORT.md in plain English from the results/ tables and the latest champion."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
RES = ROOT / "results"

PLAIN = {
    "avg_margin_capped": "average margin of victory (capped at 28)",
    "n_fbs_opp": "number of FBS opponents played",
    "n_p4_opp": "number of Power 4 opponents played",
    "sos_opp_sp": "strength of schedule (opponents' SP+)",
    "quality_wins_sp25": "wins over SP+ top-25 teams",
    "wins_vs_winning": "wins over teams above .500",
    "bad_losses": "bad losses",
    "best_win_sp_rank": "best win (SP+ rank of best team beaten)",
    "wins_vs_prev_top25": "wins over last week's committee Top 25",
    "losses": "number of losses", "wins": "number of wins", "win_pct": "win percentage",
    "sp": "SP+ rating", "sp_off": "SP+ offense", "sp_def": "SP+ defense",
    "srs": "SRS", "elo": "Elo", "fpi": "FPI",
    "off_ppa": "offensive PPA/play", "def_ppa_allowed": "defensive PPA/play allowed",
    "off_success": "offensive success rate", "def_success_allowed": "defensive success rate allowed",
    "off_explosiveness": "offensive explosiveness", "def_explosiveness_allowed": "explosive plays allowed",
    "is_p4": "Power 4 membership", "is_independent": "independent status",
    "conf_win_pct": "conference win %", "conf_games_behind": "games behind conference leader",
    "conf_contender": "in conference-title contention", "weeks_since_loss": "weeks since last loss",
    "prev_committee_rank": "last week's committee rank",
}
# Raw scale where LOWER is better: coefficient signs are flipped for display so that
# a positive number always means "helps a team's ranking".
LOWER_BETTER = {"best_win_sp_rank", "conf_games_behind", "prev_committee_rank", "sp_def",
                "def_ppa_allowed", "def_success_allowed", "def_explosiveness_allowed"}
FRAMING_NAME = {"resume": "pure résumé", "power": "pure power-rating", "combined": "combined",
                "combined_inertia": "combined + last week's rank"}


def _p(f):
    return pd.read_csv(f) if f.exists() else None


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


def write_report():
    mods = sorted((ROOT / "models").glob("week_*_champion.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not mods:
        return
    champ = joblib.load(mods[-1])
    comp, sc = _p(RES / "model_comparison.csv"), _p(RES / "scorecard.csv")
    drift, vif = _p(RES / "feature_drift.csv"), _p(RES / "collinearity.csv")
    n_weeks = champ.train_df.week.nunique()
    L = ["# Committee ranking model: REPORT", "",
         f"_Auto-generated. Current champion: `{champ.version}` (trained through release week "
         f"{champ.trained_through_week}; {n_weeks} committee release(s) of training data)._", ""]

    L += ["## 1. What the committee appears to value", ""]
    imp = champ.model.importance()
    imp = imp.mul([-1 if k in LOWER_BETTER else 1 for k in imp.index]).sort_values(key=abs, ascending=False)
    pos = [f"{PLAIN.get(k, k)} ({v:+.2f})" for k, v in imp.items() if v > 0][:5]
    neg = [f"{PLAIN.get(k, k)} ({v:+.2f})" for k, v in imp.items() if v < 0][:3]
    unit = "SHAP units" if champ.spec["family"] == "gbm" else "standardized coefficients"
    L += [f"The champion is a **{champ.spec['family']}** model on the **{FRAMING_NAME[champ.spec['framing']]}** "
          f"feature set. Largest drivers ({unit}; positive = helps a team's ranking; signs are flipped for lower-is-better stats such as SP+ rank of best win): " + "; ".join(pos) + ".",
          ("Features that work against a team: " + "; ".join(neg) + ".") if neg else "", ""]
    L += explainer_lines()
    if comp is not None:
        best = comp[comp.eligible_champion].groupby("framing").selection_score.max()
        L += ["**Résumé vs. power ratings.** Best score per philosophy (mean of Kendall tau and within-Top-25 "
              "pairwise accuracy, held-out):", ""]
        L += ["| Philosophy | Best score |", "|---|---|"]
        for fr, v in best.sort_values(ascending=False).items():
            L.append(f"| {FRAMING_NAME.get(fr, fr)} | {v:.3f} |")
        b = best.sort_values(ascending=False)
        gap = b.iloc[0] - b.iloc[1]
        L += ["", f"{FRAMING_NAME.get(b.index[0], b.index[0])} leads {FRAMING_NAME.get(b.index[1], b.index[1])} by "
                  f"{gap:.3f}; {FRAMING_NAME.get(b.index[-1], b.index[-1])} trails by {b.iloc[0] - b.iloc[-1]:.3f}. "
                  "Differences under about 0.05 are not distinguishable from noise with this little data.", ""]

    L += ["## 2. How that has shifted over the season", ""]
    if drift is None or drift.week.nunique() < 2:
        L += ["Only one committee release is in the data so far, so no shift can be measured yet. "
              "From the second release on, `results/feature_drift.csv` and `feature_drift.png` track per-week "
              "refits of the same feature set.", ""]
    else:
        P = drift[drift.model == "pairwise_coef"].pivot(index="week", columns="feature", values="value")
        chg = (P.iloc[-1] - P.iloc[0]).sort_values(key=abs, ascending=False).head(4)
        L += ["Largest changes in pairwise coefficients, first week to latest: " +
              "; ".join(f"{PLAIN.get(k, k)} {v:+.2f}" for k, v in chg.items()) +
              ". Single-week fits are noisy; treat shifts smaller than about 0.3 as noise.", ""]

    L += ["## 3. How accurate the forecasts have been", ""]
    if sc is None or not len(sc):
        L += ["No forecasts have been scored yet. The first live forecast is Week 11. "
              "Until then, the numbers in `results/model_comparison.csv` are cross-validation "
              f"({champ.cv.get('scheme', 'LOTO')}) on the Week 10 release: "
              f"Kendall tau {champ.cv.get('kendall', float('nan')):.2f}, "
              f"within-Top-25 pairwise accuracy {champ.cv.get('pairwise_acc_top25', float('nan')):.2f}, "
              f"{champ.cv.get('top25_overlap', float('nan')):.0f}/25 teams in the right Top 25 "
              f"({champ.cv.get('top12_overlap', float('nan')):.0f}/12 for the playoff field). "
              "These are optimistic: hyper-parameters and the champion were chosen on the same data.", ""]
    else:
        cols = ["week", "method", "kendall", "spearman", "pairwise_acc_top25", "top25_overlap", "top12_overlap",
                "exact_hits", "mae_rank_top25"]
        L += [sc[cols].round(3).to_markdown(index=False), "",
              "Baselines: SP+ order, and last week's committee order. The model only adds value where it beats both.", ""]
        m = sc.groupby("method")[["kendall", "pairwise_acc_top25", "top12_overlap"]].mean().round(3)
        L += ["Average across scored weeks:", "", m.to_markdown(), ""]

    L += ["## 4. Caveats (please read)", "",
          f"- **Sample size.** {n_weeks} release(s) x 138 teams. Only 25 teams per week carry real rank information; "
          "pairwise comparisons multiply rows but are not independent.",
          "- **Collinearity.** SP+, SRS, Elo, FPI and the efficiency stats overlap heavily with margin and record. "
          "Features were pruned by VIF (> 10), preferring to drop duplicative ratings first. Which of two "
          "near-identical features survives is arbitrary, so read coefficients as 'this family of information', "
          "not as causal weights."]
    if vif is not None and len(vif):
        d = vif[vif.framing == champ.spec["framing"]]
        L.append("- Dropped in the champion's framing: " + (", ".join(PLAIN.get(x, x) for x in d.dropped_feature) or "none") + ".")
    L += ["- **Selection optimism.** The champion and hyper-parameters are chosen on the same validation "
          "folds, so reported CV is optimistic. Live scored weeks are the honest test.",
          "- **Snapshot timing.** SP+/SRS/FPI/PPA have no 'as of week' option, so each week's ratings are the "
          "snapshot taken when `predict` ran. Run `predict` before Tuesday-night games.",
          "- **Release-week alignment.** Release week N uses games through CFBD week N-1.",
          "- **FCS opponents** are counted separately and excluded from SOS averages; a loss to an FCS team is a 'bad loss'.",
          "- **Bracket** conference champions are projected from current conference records (no tiebreakers, "
          "no title games). Seeding rules are assumed.",
          "- **Circularity.** 'Wins vs. committee Top 25' only uses *last* week's committee list (not available at Week 10).",
          ""]
    (ROOT / "REPORT.md").write_text("\n".join(L))

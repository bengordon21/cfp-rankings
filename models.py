"""Models for the committee ranking: pointwise, pairwise (primary), gradient boosting.

Conventions
- Target: committee rank 1..25; every unranked FBS team = 26 (a tie).
- Every model produces a per-team `score` (higher = ranked better).
- Features are z-scored within each release week across all 138 FBS teams.
- Validation: leave-one-team-out (1 week of data) or leave-one-week-out (2+ weeks).
"""
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

import build_features as bf

UNRANKED = 26
RESUME = ["wins", "losses", "win_pct", "avg_margin_capped", "n_fbs_opp", "n_p4_opp",
          "sos_opp_sp", "quality_wins_sp25", "wins_vs_winning", "bad_losses",
          "best_win_sp_rank", "wins_vs_prev_top25"]
POWER = ["sp", "sp_off", "sp_def", "srs", "elo", "fpi", "off_ppa", "def_ppa_allowed",
         "off_success", "def_success_allowed", "off_explosiveness", "def_explosiveness_allowed"]
CONTEXT = ["is_p4", "is_independent", "conf_win_pct", "conf_games_behind", "conf_contender",
           "weeks_since_loss"]
FRAMINGS = {
    "resume": RESUME,
    "power": POWER,
    "combined": RESUME + POWER + CONTEXT,
    "combined_inertia": RESUME + POWER + CONTEXT + ["prev_committee_rank"],  # needs weeks >= 11
}
# Collinearity pruning prefers to drop duplicative power-rating / efficiency features before
# committee-style record & resume features, so the model stays interpretable. At each step:
# among features with VIF > VIF_MAX, drop the first one found in this list; if none, drop max-VIF.
DROP_PRIORITY = ["win_pct", "wins", "sp_off", "sp_def", "fpi", "srs", "elo", "off_ppa",
                 "def_ppa_allowed", "off_success", "def_success_allowed", "off_explosiveness",
                 "def_explosiveness_allowed", "n_fbs_opp", "is_p4", "conf_win_pct"]
VIF_MAX = 10.0
GRIDS = {"pointwise": [0.1, 0.5, 1.5], "pairwise": [0.003, 0.01, 0.05], "gbm": [None]}


# ----------------------------------------------------------------------------- data
def load_training(actuals, weeks=None):
    """Training frame: one row per FBS team per week with z-scored features and rank.

    `actuals` is a DataFrame(week, team, rank) of the committee Top 25s we know.
    Returns (df, h2h) where h2h = {week: set((winner, loser))}.
    """
    weeks = sorted(actuals.week.unique()) if weeks is None else weeks
    frames, h2h = [], {}
    for w in weeks:
        X, h = bf.get_features(w)
        if X.prev_committee_rank.notna().any():
            X["prev_committee_rank"] = X.prev_committee_rank.fillna(UNRANKED)
        Z = bf.zscore(X)
        a = actuals[actuals.week == w].set_index("team")["rank"]
        Z["rank"] = a.reindex(Z.index).fillna(UNRANKED).astype(float)
        Z["week"] = w
        Z["team"] = Z.index
        frames.append(Z.reset_index(drop=True))
        h2h[w] = set(zip(h.winner, h.loser))
    return pd.concat(frames, ignore_index=True), h2h


def usable_feats(df, feats):
    """Drop features that are constant across the whole training frame (e.g. no prev rank yet)."""
    return [f for f in feats if f in df and df[f].std() > 1e-9]


def select_features(df, feats, framing=""):
    """Iteratively drop the feature with the largest VIF until all VIF <= VIF_MAX.

    Returns (kept, log) where log lists (framing, dropped, vif_at_drop).
    """
    feats = usable_feats(df, feats)
    log = []
    while len(feats) > 2:
        M = df[feats].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = np.array([variance_inflation_factor(M, i) for i in range(len(feats))])
        v = np.where(np.isfinite(v), v, 1e12)
        if v.max() <= VIF_MAX:
            break
        high = [f for f, x in zip(feats, v) if x > VIF_MAX]
        pick = next((f for f in DROP_PRIORITY if f in high), feats[int(v.argmax())])
        log.append((framing, pick, float(v[feats.index(pick)])))
        feats = [f for f in feats if f != pick]
    return feats, log


# ----------------------------------------------------------------------------- models
def _pairs(df, feats, h2h, with_h2h=True):
    """All (i above j) pairs within each week; returns X (diffs + h2h col), y, weights."""
    Xs, ys = [], []
    for w, g in df.groupby("week"):
        Z, r, teams = g[feats].values, g["rank"].values, g["team"].values
        i, j = np.where(r[:, None] < r[None, :])
        d = Z[i] - Z[j]
        hh = h2h.get(w, set())
        h = np.array([(1.0 if (teams[a], teams[b]) in hh else (-1.0 if (teams[b], teams[a]) in hh else 0.0))
                      for a, b in zip(i, j)])
        D = np.column_stack([d, h])
        Xs += [D, -D]
        ys += [np.ones(len(i)), np.zeros(len(i))]
    return np.vstack(Xs), np.concatenate(ys)


class Model:
    """One fitted ranking model. family in {pointwise, pairwise, gbm}."""

    def __init__(self, family, param, feats, seed=0):
        self.family, self.param, self.feats, self.seed = family, param, list(feats), seed
        self.beta = 0.0      # h2h coefficient (pairwise only)
        self.coef = None

    def fit(self, df, h2h):
        F = self.feats
        if self.family == "pointwise":
            self.m = ElasticNet(alpha=self.param, l1_ratio=0.5, max_iter=5000).fit(df[F].values, df["rank"].values)
            self.coef = -self.m.coef_            # goodness orientation: positive = ranks higher
        elif self.family == "pairwise":
            X, y = _pairs(df, F, h2h)
            self.m = LogisticRegression(C=self.param, solver="liblinear", fit_intercept=False,
                                        l1_ratio=1.0, max_iter=2000).fit(X, y)
            self.coef = self.m.coef_[0][:-1]
            self.beta = float(self.m.coef_[0][-1])
        else:
            self.m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                               subsample=0.8, min_samples_leaf=3, random_state=self.seed
                                               ).fit(df[F].values, df["rank"].values)
        return self

    def base_score(self, df):
        if self.family == "gbm":
            return -self.m.predict(df[self.feats].values)
        return df[self.feats].values @ self.coef

    def score(self, df, h2h_w=None, beta=None):
        """Per-team score for ONE week's rows, with the head-to-head adjustment for pairwise."""
        s = self.base_score(df)
        b = self.beta if beta is None else beta
        if self.family == "pairwise" and b and h2h_w:
            s = h2h_adjust(s, df["team"].values, h2h_w, b)
        return s

    def contributions(self, df):
        """Per-team, per-feature contribution to 'goodness' vs the average FBS team."""
        if self.family == "gbm":
            import shap
            sv = shap.TreeExplainer(self.m).shap_values(df[self.feats].values)
            return pd.DataFrame(-sv, index=df["team"].values, columns=self.feats)
        return pd.DataFrame(df[self.feats].values * self.coef, index=df["team"].values, columns=self.feats)

    def importance(self):
        if self.family == "gbm":
            return pd.Series(np.abs(self.contributions(self._train)).mean().values, index=self.feats)
        return pd.Series(self.coef, index=self.feats)


def h2h_adjust(s, teams, hh, beta, window=40, passes=10):
    """Swap adjacent teams when a head-to-head result outweighs the score gap (beta*h2h)."""
    order = list(np.argsort(-s))
    sorted_scores = np.array(sorted(s, reverse=True))
    for _ in range(passes):
        changed = False
        for k in range(min(window, len(order) - 1)):
            a, b = order[k], order[k + 1]
            h = 1.0 if (teams[a], teams[b]) in hh else (-1.0 if (teams[b], teams[a]) in hh else 0.0)
            if (s[a] - s[b]) + beta * h < 0:   # P(a above b) < .5 after head-to-head
                order[k], order[k + 1] = b, a
                changed = True
        if not changed:
            break
    out = np.empty_like(s)
    for pos, t in enumerate(order):
        out[t] = sorted_scores[pos]
    return out


# ----------------------------------------------------------------------------- metrics
def week_metrics(rank, score):
    """rank: actual (26 = unranked tie); score: higher = better. Returns dict of metrics."""
    rank, score = np.asarray(rank, float), np.asarray(score, float)
    pred_rank = (-score).argsort().argsort() + 1.0
    top = rank <= 25
    diff = rank[:, None] != rank[None, :]
    above = rank[:, None] < rank[None, :]
    correct = (score[:, None] > score[None, :]) == above
    both = top[:, None] & top[None, :] & diff
    pred_top = set(np.where(pred_rank <= 25)[0]); act_top = set(np.where(top)[0])
    pred_12 = set(np.where(pred_rank <= 12)[0]); act_12 = set(np.where(rank <= 12)[0])
    return {
        "spearman": spearmanr(rank, pred_rank)[0],
        "kendall": kendalltau(rank, pred_rank)[0],
        "pairwise_acc": correct[diff].mean(),
        "pairwise_acc_top25": correct[both].mean(),
        "top25_overlap": len(pred_top & act_top),
        "top12_overlap": len(pred_12 & act_12),
        "exact_hits": int((pred_rank[top] == rank[top]).sum()),
        "mae_rank_top25": float(np.abs(pred_rank[top] - rank[top]).mean()),
    }


# ----------------------------------------------------------------------------- validation
def cv_scores(df, h2h, spec, feats):
    """Held-out scores: LOWO if >=2 weeks else LOTO. Returns df[week, team, rank, score]."""
    weeks = sorted(df.week.unique())
    out = []
    if len(weeks) >= 2:
        for w in weeks:
            m = Model(spec["family"], spec["param"], feats).fit(df[df.week != w], h2h)
            te = df[df.week == w]
            out.append(te[["week", "team", "rank"]].assign(score=m.score(te, h2h.get(w))))
    else:
        w = weeks[0]
        s, betas = {}, []
        for t in df.team:
            m = Model(spec["family"], spec["param"], feats).fit(df[df.team != t], h2h)
            te = df[df.team == t]
            s[t] = m.base_score(te)[0]
            betas.append(m.beta)
        te = df.copy()
        sc = np.array([s[t] for t in te.team])
        if spec["family"] == "pairwise":
            sc = h2h_adjust(sc, te.team.values, h2h.get(w, set()), float(np.mean(betas)))
        out.append(te[["week", "team", "rank"]].assign(score=sc))
    return pd.concat(out, ignore_index=True)


def evaluate_candidates(df, h2h, log=print):
    """Run every (framing x family x hyper-parameter) candidate; returns (per_week, summary, vif_log)."""
    weeks = sorted(df.week.unique())
    rows, vif_log, kept = [], [], {}
    for fr, feats in FRAMINGS.items():
        sub, ws = df, weeks
        if fr == "combined_inertia":
            ws = [w for w in weeks if w >= 11]
            if not ws:
                continue
            sub = df[df.week.isin(ws)]
        k, lg = select_features(sub, feats, fr)
        kept[fr] = k; vif_log += lg
        for fam, grid in GRIDS.items():
            for p in grid:
                spec = {"framing": fr, "family": fam, "param": p}
                cv = cv_scores(sub, h2h, spec, k)
                for w, g in cv.groupby("week"):
                    rows.append({**spec, "week": w, "n_features": len(k), **week_metrics(g["rank"], g["score"])})
                log(f"  evaluated {fr}/{fam}/{p}")
    per_week = pd.DataFrame(rows)
    return per_week, kept, pd.DataFrame(vif_log, columns=["framing", "dropped_feature", "vif_at_drop"])


def summarize(per_week, weeks):
    """Mean metrics per candidate + selection score on the common set of weeks."""
    key = ["framing", "family", "param"]
    inertia = per_week[per_week.framing == "combined_inertia"]
    eligible_inertia = inertia.week.nunique() >= 2
    pool = per_week if eligible_inertia else per_week[per_week.framing != "combined_inertia"]
    common = sorted(set(pool.groupby("framing").week.apply(set).agg(lambda s: set.intersection(*s))))
    P = pool[pool.week.isin(common)]
    S = P.groupby(key, dropna=False).mean(numeric_only=True).drop(columns=["week"]).reset_index()
    S["n_weeks_scored"] = len(common)
    S["selection_score"] = (S.kendall + S.pairwise_acc_top25) / 2
    S["scheme"] = "LOWO" if len(weeks) >= 2 else "LOTO"
    S["eligible_champion"] = True
    if not eligible_inertia and len(inertia):
        extra = inertia.groupby(key, dropna=False).mean(numeric_only=True).drop(columns=["week"]).reset_index()
        extra["n_weeks_scored"] = inertia.week.nunique()
        extra["selection_score"] = (extra.kendall + extra.pairwise_acc_top25) / 2
        extra["scheme"] = "LOTO(diagnostic)"; extra["eligible_champion"] = False
        S = pd.concat([S, extra], ignore_index=True)
    return S.sort_values("selection_score", ascending=False).reset_index(drop=True)


def choose_champion(S):
    """Best selection_score; prefer pairwise (primary) if within 0.01 of the best."""
    E = S[S.eligible_champion]
    best = E.selection_score.max()
    close = E[E.selection_score >= best - 0.01]
    pw = close[close.family == "pairwise"]
    return (pw if len(pw) else close).iloc[0].to_dict()


# ----------------------------------------------------------------------------- champion
@dataclass
class Champion:
    spec: dict
    feats: list
    model: Model
    train_df: pd.DataFrame
    h2h: dict
    trained_through_week: int
    version: str
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    cv: dict = field(default_factory=dict)


def fit_champion(spec, kept, df, h2h, week, cv=None):
    feats = kept[spec["framing"]]
    sub = df if spec["framing"] != "combined_inertia" else df[df.week >= 11]
    m = Model(spec["family"], spec["param"], feats).fit(sub, h2h)
    m._train = sub
    p = "" if spec["param"] is None or (isinstance(spec["param"], float) and np.isnan(spec["param"])) else f"-{spec['param']}"
    return Champion(spec, feats, m, sub, h2h, int(week), f"w{week}-{spec['family']}-{spec['framing']}{p}", cv=cv or {})


def bootstrap_ranks(champ, test_df, h2h_w, B=200, seed=0):
    """Team-level bootstrap: refit on resampled teams (within each week), re-rank test teams."""
    rng = np.random.default_rng(seed)
    base = champ.train_df
    R = np.zeros((B, len(test_df)))
    for b in range(B):
        parts = []
        for w, g in base.groupby("week"):
            parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        bs = pd.concat(parts, ignore_index=True)
        if champ.spec["family"] == "pairwise":
            bs = bs.drop_duplicates(["week", "team"])
        m = Model(champ.spec["family"], champ.spec["param"], champ.feats, seed=b).fit(bs, champ.h2h)
        R[b] = (-m.score(test_df, h2h_w)).argsort().argsort() + 1
    return R

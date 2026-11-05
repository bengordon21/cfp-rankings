"""Build per-team features as of a committee release (see FEATURES.md).

Release week N uses ONLY games with CFBD week <= N-1 (the committee's "Week 10"
show airs Tuesday after CFBD week 9 games). Ratings / efficiency come from snapshots
stored under the release stamp (data/raw/*_rel{N}.json), so history never changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
FEAT_DIR = ROOT / "data" / "features"
P4 = {"SEC", "Big Ten", "Big 12", "ACC"}
MARGIN_CAP = 28
BAD_LOSS_SP_RANK = 40        # loss to a team outside SP+ top 40 counts as "bad"
NO_WIN_RANK = 139            # best_win_sp_rank when a team has no FBS win


def _load(name):
    return json.loads((RAW / f"{name}.json").read_text())


def committee_ranks(week):
    """{team: rank} from the stored committee poll, or None if not available."""
    f = RAW / f"cfp_rankings_wk{week}.json"
    if not f.exists():
        return None
    for w in json.loads(f.read_text()):
        if w["week"] != week:
            continue
        for p in w["polls"]:
            if p["poll"] == "Playoff Committee Rankings":
                return {r["school"]: r["rank"] for r in p["ranks"]}
    return None


def _long_games(games, fbs, n):
    g = games[(games.week <= n - 1) & games.completed & games.homePoints.notna()].copy()
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "team": g[f"{side}Team"], "opp": g[f"{other}Team"], "week": g.week,
            "pf": g[f"{side}Points"], "pa": g[f"{other}Points"],
            "opp_conf": g[f"{other}Conference"], "conf_game": g.conferenceGame,
        })
        rows.append(d)
    L = pd.concat(rows)
    L = L[L.team.isin(fbs)].copy()
    L["opp_fbs"] = L.opp.isin(fbs)
    L["win"] = L.pf > L.pa
    L["margin"] = L.pf - L.pa
    return L


def build_raw_features(n):
    """Unscaled feature table (index=team) plus h2h list for release week n."""
    teams = pd.DataFrame(_load("teams_fbs")).set_index("school")
    fbs = set(teams.index)
    games = pd.DataFrame(_load(f"games_rel{n}"))
    L = _long_games(games, fbs, n)

    sp = pd.DataFrame(_load(f"ratings_sp_rel{n}"))
    sp = sp[sp.team.isin(fbs)].set_index("team")
    srs = pd.DataFrame(_load(f"ratings_srs_rel{n}"))
    # CFBD lists a few names twice (e.g. Charlotte, Troy) with one conference-less row; keep the conferenced one
    srs = srs.sort_values("conference", na_position="last").drop_duplicates("team").set_index("team")
    elo = pd.DataFrame(_load(f"ratings_elo_rel{n}")).set_index("team")
    fpi = pd.DataFrame(_load(f"ratings_fpi_rel{n}")).set_index("team")
    adv = {r["team"]: r for r in _load(f"stats_adv_rel{n}")}

    X = pd.DataFrame(index=sorted(fbs))
    X["conference"] = teams.conference
    X["sp"] = sp.rating
    X["sp_off"] = sp.offense.map(lambda d: d["rating"])
    X["sp_def"] = sp.defense.map(lambda d: d["rating"])         # lower = better
    X["srs"] = srs.rating.reindex(X.index)
    X["elo"] = elo.elo.reindex(X.index)
    X["fpi"] = fpi.fpi.reindex(X.index)
    X["off_ppa"] = [adv[t]["offense"]["ppa"] for t in X.index]
    X["def_ppa_allowed"] = [adv[t]["defense"]["ppa"] for t in X.index]
    X["off_success"] = [adv[t]["offense"]["successRate"] for t in X.index]
    X["def_success_allowed"] = [adv[t]["defense"]["successRate"] for t in X.index]
    X["off_explosiveness"] = [adv[t]["offense"]["explosiveness"] for t in X.index]
    X["def_explosiveness_allowed"] = [adv[t]["defense"]["explosiveness"] for t in X.index]
    missing = X.drop(columns="conference").isna().sum()
    if missing.any():
        print("WARNING missing feature values:\n", missing[missing > 0])

    sp_rank = X.sp.rank(ascending=False, method="min")
    top25_sp = set(sp_rank[sp_rank <= 25].index)

    rec = L.groupby("team").agg(games=("win", "size"), wins=("win", "sum"))
    rec["losses"] = rec.games - rec.wins
    rec["win_pct"] = rec.wins / rec.games
    wp = rec.win_pct.to_dict()

    L["margin_c"] = L.margin.clip(-MARGIN_CAP, MARGIN_CAP)
    L["opp_sp"] = L.opp.map(X.sp)
    L["opp_sp_rank"] = L.opp.map(sp_rank)
    L["opp_wp"] = L.opp.map(wp)
    L["opp_p4"] = L.opp_fbs & L.opp.map(X.conference).isin(P4)

    f = pd.DataFrame(index=X.index)
    f["wins"] = rec.wins; f["losses"] = rec.losses; f["win_pct"] = rec.win_pct
    f["avg_margin_capped"] = L.groupby("team").margin_c.mean()
    fb = L[L.opp_fbs]
    f["n_fbs_opp"] = fb.groupby("team").size()
    f["n_p4_opp"] = L.groupby("team").opp_p4.sum()
    f["n_fcs_opp"] = L[~L.opp_fbs].groupby("team").size()
    f["sos_opp_sp"] = fb.groupby("team").opp_sp.mean()
    wins = fb[fb.win]
    losses = L[~L.win]
    f["quality_wins_sp25"] = wins[wins.opp.isin(top25_sp)].groupby("team").size()
    f["wins_vs_winning"] = wins[wins.opp_wp > 0.5].groupby("team").size()
    bad = losses[(~losses.opp_fbs) | (losses.opp_wp < 0.5) | (losses.opp_sp_rank > BAD_LOSS_SP_RANK)]
    f["bad_losses"] = bad.groupby("team").size()
    f["best_win_sp_rank"] = wins.groupby("team").opp_sp_rank.min()
    prev = committee_ranks(n - 1)
    f["prev_committee_rank"] = pd.Series(prev).reindex(f.index) if prev else np.nan
    pc = set(prev) if prev else set()
    f["wins_vs_prev_top25"] = wins[wins.opp.isin(pc)].groupby("team").size() if prev else 0

    # context
    f["is_p4"] = X.conference.isin(P4).astype(float)
    f["is_independent"] = (X.conference == "FBS Independents").astype(float)
    cg = L[L.conf_game & L.opp_fbs]
    cr = cg.groupby("team").agg(cw=("win", "sum"), cn=("win", "size"))
    cr["cl"] = cr.cn - cr.cw
    f["conf_wins"] = cr.cw; f["conf_losses"] = cr.cl
    f = f.fillna({"n_fbs_opp": 0, "n_p4_opp": 0, "n_fcs_opp": 0, "quality_wins_sp25": 0,
                  "wins_vs_winning": 0, "bad_losses": 0, "conf_wins": 0, "conf_losses": 0,
                  "wins_vs_prev_top25": 0})
    f["best_win_sp_rank"] = f.best_win_sp_rank.fillna(NO_WIN_RANK)
    f["conf_win_pct"] = (f.conf_wins / (f.conf_wins + f.conf_losses)).fillna(f.win_pct)
    f["conference"] = X.conference
    lead = {}
    for c, grp in f.groupby("conference"):
        if c == "FBS Independents":
            continue
        top = grp.assign(d=grp.conf_wins - grp.conf_losses).sort_values(["d", "conf_wins"], ascending=False).iloc[0]
        lead[c] = (top.conf_wins, top.conf_losses)
    def behind(r):
        if r.conference not in lead:
            return 0.0
        lw, ll = lead[r.conference]
        return ((lw - r.conf_wins) + (r.conf_losses - ll)) / 2
    f["conf_games_behind"] = f.apply(behind, axis=1)
    f["conf_contender"] = ((f.conf_games_behind <= 1) & (f.is_independent == 0)).astype(float)
    last_loss = losses.groupby("team").week.max()
    f["weeks_since_loss"] = (n - 1 - last_loss).reindex(f.index).fillna(n)
    f = f.drop(columns=["conference"])
    X = X.join(f)

    h2h = L[L.opp_fbs & L.win][["team", "opp"]].rename(columns={"team": "winner", "opp": "loser"})
    return X, h2h.reset_index(drop=True)


def get_features(n, rebuild=False):
    """Cached raw features + h2h for release n (stored so history is frozen)."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    fx, fh = FEAT_DIR / f"features_wk{n}.csv", FEAT_DIR / f"h2h_wk{n}.csv"
    if fx.exists() and fh.exists() and not rebuild:
        return pd.read_csv(fx, index_col=0), pd.read_csv(fh)
    X, h = build_raw_features(n)
    X.to_csv(fx); h.to_csv(fh, index=False)
    return X, h


def zscore(X):
    """Per-week z-scores across all FBS teams (numeric columns); constants -> 0."""
    num = X.select_dtypes("number")
    sd = num.std(ddof=0).replace(0, np.nan)
    return ((num - num.mean()) / sd).fillna(0.0)


if __name__ == "__main__":
    import sys
    X, h = get_features(int(sys.argv[1]), rebuild=True)
    print(X.shape, len(h))
    print(X.sort_values("sp", ascending=False).head(12).round(2).T.to_string())

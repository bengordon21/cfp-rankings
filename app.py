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
    if "sign_conflict" not in w:                      # files written before the standalone-corr column existed
        w["sign_conflict"], w["corr"] = False, float("nan")
    view = st.radio("View", ["Individual factors", "Groups"], horizontal=True, key="valued_view")
    if view == "Groups":
        g = w.groupby("group").influence.sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h"))
        fig.update_layout(xaxis_title="Share of influence", height=350)
    else:
        w = w.assign(name=w.factor.map(nm)).sort_values("weight")
        fig = go.Figure(go.Bar(
            x=w.weight, y=w.name, orientation="h",
            marker_color=["#e67e22" if c else "#2e86c1" for c in w.sign_conflict],
            customdata=w["corr"], hovertemplate="%{y}<br>weight %{x:+.2f}<br>standalone corr %{customdata:+.2f}<extra></extra>",
            error_x=dict(type="data", symmetric=False, array=(w.hi - w.weight).clip(lower=0),
                         arrayminus=(w.weight - w.lo).clip(lower=0))))
        fig.update_layout(xaxis_title="Weight (rank-points per 1 SD); bars = 90% bootstrap band",
                          height=max(350, 28 * len(w)))
    st.plotly_chart(fig)
    st.caption("Positive = a higher value of the stat helps a team; negative = a higher value hurts. "
               "Orange bars: the weight's sign contradicts the stat's standalone correlation with the committee's "
               "order (stats overlap, so read those as 'given the other stats'). "
               "One week has only 25 ranked teams, so lean on the bands and the pooled fit below.")
    if view == "Individual factors":
        tbl = w.assign(factor=w.factor.map(nm), flag=w.sign_conflict.map({True: "⚠ sign differs", False: ""}))
        st.dataframe(tbl.sort_values("influence", ascending=False)[
            ["factor", "group", "weight", "lo", "hi", "corr", "flag", "influence"]
        ].rename(columns={"corr": "standalone corr", "lo": "band low", "hi": "band high"}), hide_index=True)
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

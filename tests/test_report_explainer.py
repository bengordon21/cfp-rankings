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

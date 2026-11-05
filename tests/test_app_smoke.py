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

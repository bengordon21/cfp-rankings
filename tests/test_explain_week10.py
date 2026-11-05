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

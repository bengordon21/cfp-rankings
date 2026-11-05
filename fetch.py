"""CFBD API access with disk caching and call logging.

Rules: every response is cached to data/raw/<name>.json and never re-fetched
(unless force=True). Every real network call is appended to logs/api_calls.csv.
The API key is read from CFBD_API_KEY (via .env) and is never printed or logged.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
LOG = ROOT / "logs" / "api_calls.csv"
BASE = "https://api.collegefootballdata.com"
YEAR = 2026

load_dotenv(ROOT / ".env")


def _key():
    k = os.environ.get("CFBD_API_KEY", "").strip()
    if not k or k == "your_key_here":
        sys.exit("CFBD_API_KEY is not set. Put it in .env (see .env.example).")
    return k


def _log(name, path, status, cached, note=""):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "cache_name", "path", "status", "network_call", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    name, path, status, 0 if cached else 1, note])


def get(name, path, params=None, force=False):
    """Return parsed JSON for `path`, caching at data/raw/<name>.json.

    Failed responses (non-200) are NOT cached; the status is logged and an
    exception is raised so nothing is silently guessed.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{name}.json"
    if f.exists() and not force:
        _log(name, path, "cache", True)
        return json.loads(f.read_text())
    r = requests.get(BASE + path, params=params,
                     headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    _log(name, path, r.status_code, False, note=str(params or {}))
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    f.write_text(json.dumps(data))
    time.sleep(0.3)
    return data


def info():
    """Quota status. Not cached (it changes); may itself count as a call."""
    r = requests.get(BASE + "/info", headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    _log("info", "/info", r.status_code, False)
    return r.json() if r.status_code == 200 else {"status": r.status_code}


def fetch_release(n):
    """All data needed to build features for committee release week n.

    Games through CFBD week n-1 only matter, but /games is one season-level call,
    so we pull once per release and filter by week in build_features. Ratings have
    no week parameter (live snapshots), so each is stored under a release stamp.
    ~7 network calls per new release; 0 if everything is cached.
    """
    get("teams_fbs", "/teams/fbs", {"year": YEAR})
    get("conferences", "/conferences")
    get(f"games_rel{n}", "/games", {"year": YEAR, "seasonType": "regular"})
    for r in ("sp", "srs", "elo", "fpi"):
        get(f"ratings_{r}_rel{n}", f"/ratings/{r}", {"year": YEAR})
    get(f"stats_adv_rel{n}", "/stats/season/advanced",
        {"year": YEAR, "endWeek": n - 1, "excludeGarbageTime": "true"})


def fetch_committee(n, force=False):
    """Committee poll for release week n (the 'Playoff Committee Rankings' poll)."""
    return get(f"cfp_rankings_wk{n}", "/rankings",
               {"year": YEAR, "poll": "cfp", "week": n, "seasonType": "regular"}, force=force)


def fetch_polls(n):
    """AP / Coaches polls (comparison only, never a feature)."""
    return get(f"polls_all_wk{n}", "/rankings", {"year": YEAR, "seasonType": "regular"})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "info":
        i = info()
        print({k: i.get(k) for k in ("tierName", "monthlyLimit", "remainingCalls", "usedCalls", "resetAt")})
    elif cmd == "release" and len(sys.argv) > 2:
        fetch_release(int(sys.argv[2]))
        print("remaining calls:", info().get("remainingCalls"))

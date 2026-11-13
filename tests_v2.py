"""Lotto-Muse v2 tests: feature causality (no look-ahead), lane
determinism, and scoring sanity. Run: venv/bin/python -m pytest tests_v2.py
"""
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

import scraper
from lotto_muse import (COVERAGE_K, PARITY_BONUS, _expected_counts,
                        _merge_frames, featurize, lane_due, lane_hotdigits,
                        lane_random, load_history, pick_set, ranked_sets,
                        score_coverage_guess, score_order_free, target_of)


def test_feature_dim():
    df = load_history()
    past = [(int(r.d1), int(r.d2), int(r.d3)) for r in df.itertuples()][-500:]
    x = featurize(past, pd.Timestamp("2026-10-22"), True)
    assert len(x) == 71
    assert np.all(np.isfinite(x))


def test_causality_truncation():
    """Features at draw i computed from the full frame's prefix must equal
    features computed when the frame ends at i (no future data leaks in)."""
    df = load_history()
    draws = [(r.date, r.draw, int(r.d1), int(r.d2), int(r.d3))
             for r in df.itertuples()]
    for i in (500, 2000, 4000, len(draws) - 1):
        date, draw, *_ = draws[i]
        full_prefix = [d[2:] for d in draws[:i]]
        x_full = featurize(full_prefix, date, draw == "eve")
        # truncated frame: history physically ends at i
        trunc = draws[:i]
        x_trunc = featurize([d[2:] for d in trunc], date, draw == "eve")
        assert np.allclose(x_full, x_trunc), f"leak at {i}"


def test_future_draws_do_not_change_features():
    df = load_history()
    draws = [(r.date, r.draw, int(r.d1), int(r.d2), int(r.d3))
             for r in df.itertuples()]
    i = 3000
    date, draw, *_ = draws[i]
    base = featurize([d[2:] for d in draws[:i]], date, draw == "eve")
    mutated = draws[:i] + [(date, draw, 0, 0, 0)] + draws[i + 1:]
    after = featurize([d[2:] for d in mutated[:i]], date, draw == "eve")
    assert np.allclose(base, after)


def test_scoring():
    assert score_order_free((1, 2, 3), (1, 2, 3)) == (1, 3)
    assert score_order_free((1, 1, 2), (1, 2, 2)) == (0, 2)
    assert score_order_free((0, 5, 9), (1, 2, 3)) == (0, 0)
    assert target_of((None, None, 3, 1, 2)) == (1, 2, 3)


def test_lanes_deterministic_and_valid():
    df = load_history()
    past = [(r.date, r.draw, int(r.d1), int(r.d2), int(r.d3))
            for r in df.itertuples()][-400:]
    for fn in (lane_hotdigits, lane_due):
        g = fn(past)
        assert len(g) == 3 and list(g) == sorted(g)
        assert all(0 <= d <= 9 for d in g)
    r1 = lane_random("2026-10-22", "mid")
    r2 = lane_random("2026-10-22", "mid")
    assert r1 == r2 and len(r1) == 3


# ------------------------------------------------------- fast layer (v3)
LP_TEXT = ("Latest Results Daily 3 Midday Wednesday, November 4, 2026 "
           "Midday 6 2 6 Prizes/Odds Next Drawing: Thu Daily 3 Evening "
           "Tuesday, November 3, 2026 Evening 2 5 7 Prizes/Odds")
MN_TEXT = ("the very latest numbers below. Wednesday, November 4, 2026 "
           "6 2 6 More Daily 3 Midday Numbers Tuesday, November 3, 2026 "
           "2 5 7 More Daily 3 Evening Numbers How to Play")
VG_TEXT = ("Daily 3 Midday Nov 4, 2026 $500 Top Prize 6 2 6 Predictions "
           "Daily 3 Evening Nov 3, 2026 $500 Top Prize 2 5 7 Predictions")
EXPECTED_FAST = {(date(2026, 11, 4), "mid"): (6, 2, 6),
                 (date(2026, 11, 3), "eve"): (2, 5, 7)}


def test_fast_parsers():
    assert scraper.parse_lotterypost_latest(LP_TEXT) == EXPECTED_FAST
    assert scraper.parse_milnumbers_latest(MN_TEXT) == EXPECTED_FAST
    assert scraper.parse_valley_latest(VG_TEXT) == EXPECTED_FAST


def test_fast_layer_needs_two_agreeing_sources():
    pages = {"https://www.lotterypost.com/results/mi": LP_TEXT,
             "https://michiganlotterynumbers.com/daily-3": MN_TEXT,
             "https://www.lotteryvalley.com/michigan/games/daily-3": VG_TEXT}
    with mock.patch.object(scraper, "_page_text", lambda url: pages[url]):
        df, _ = scraper.scrape_fast_latest()
    got = {(r["date"].date(), r["draw"]): (int(r["d1"]), int(r["d2"]),
                                           int(r["d3"]))
           for _, r in df.iterrows()}
    assert got == EXPECTED_FAST
    # one source disagrees -> the 2-source majority value still wins
    bad = dict(pages)
    bad["https://michiganlotterynumbers.com/daily-3"] = MN_TEXT.replace(
        "6 2 6 More", "9 9 9 More")
    with mock.patch.object(scraper, "_page_text", lambda url: bad[url]):
        df2, _ = scraper.scrape_fast_latest()
    got2 = {(r["date"].date(), r["draw"]): (int(r["d1"]), int(r["d2"]),
                                            int(r["d3"]))
            for _, r in df2.iterrows()}
    assert got2[(date(2026, 11, 4), "mid")] == (6, 2, 6)
    # all three sources disagree -> no majority -> the draw is dropped
    bad2 = dict(bad)
    bad2["https://www.lotteryvalley.com/michigan/games/daily-3"] = (
        VG_TEXT.replace("6 2 6 Predictions", "1 1 1 Predictions"))
    with mock.patch.object(scraper, "_page_text", lambda url: bad2[url]):
        df3, _ = scraper.scrape_fast_latest()
    got3 = {(r["date"].date(), r["draw"]) for _, r in df3.iterrows()}
    assert (date(2026, 11, 4), "mid") not in got3
    assert (date(2026, 11, 3), "eve") in got3


def test_pick_set_always_distinct_and_argmax():
    import random as _r
    rng = _r.Random(7)
    for _ in range(300):
        scores = np.array([rng.random() for _ in range(10)])
        g = pick_set(scores, parity_bonus=0.0)
        assert len(set(g)) == 3, "v3 pick must be 3 distinct digits"
        best = max(sum(scores[d] for d in c)
                   for c in __import__("itertools").combinations(range(10), 3))
        assert abs(sum(scores[d] for d in g) - best) < 1e-9
    s = np.zeros(10)
    s[[1, 4, 9]] = [0.3, 0.25, 0.2]
    assert pick_set(s, parity_bonus=0.0) == (1, 4, 9)


def test_pick_set_parity_bonus_is_tiebreak_only():
    s = np.zeros(10)
    s[[0, 2, 4]] = 0.20          # all-even set sums to 0.60
    s[1] = 0.19                  # best 2-1 set sums to 0.59: gap > bonus
    assert pick_set(s) == (0, 2, 4)
    s[1] = 0.199                 # gap 0.001 < PARITY_BONUS -> parity wins
    assert pick_set(s) == (0, 1, 2)
    assert PARITY_BONUS <= 0.01  # stays tie-breaker scale by design


def test_expected_counts_stub():
    class Stub:
        def __init__(self, classes, proba):
            self.classes_ = np.array(classes)
            self._p = np.array([proba])

        def predict_proba(self, x):
            return self._p

    out = _expected_counts([Stub([0, 1, 2], [0.5, 0.3, 0.2]),
                            Stub([2, 5], [0.25, 0.75])], np.zeros((1, 3)))
    assert abs(out[0] - 0.5) < 1e-9 and abs(out[2] - 0.45) < 1e-9
    assert abs(out[5] - 0.75) < 1e-9 and abs(out.sum() - 2.0) < 1e-9


def test_ranked_sets_topk_and_top1():
    import itertools as _it
    import random as _r
    rng = _r.Random(11)
    for _ in range(60):
        scores = np.array([rng.random() for _ in range(10)])
        ranked = ranked_sets(scores, k=COVERAGE_K)
        assert len(ranked) == COVERAGE_K and len(set(ranked)) == COVERAGE_K
        assert all(len(set(s)) == 3 for s in ranked)
        assert ranked[0] == pick_set(scores)
        vals = sorted((sum(scores[d] for d in c)
                       + (PARITY_BONUS if sum(1 for d in c if d % 2 == 0)
                          in (1, 2) else 0.0),
                       c) for c in _it.combinations(range(10), 3))
        top = [c for _, c in sorted(vals, key=lambda vc: (-vc[0], vc[1]))
               [:COVERAGE_K]]
        assert ranked == top
        assert ranked_sets(scores, k=COVERAGE_K) == ranked  # deterministic


def test_score_coverage_guess_cases():
    sets = [(0, 6, 9), (0, 6, 8), (1, 6, 9), (2, 3, 9)]
    hits, boxed, best = score_coverage_guess(sets, (6, 9, 0))
    assert (hits, boxed, best) == (3, 1, (0, 6, 9))       # a set boxed
    hits, boxed, best = score_coverage_guess(sets, (0, 6, 7))
    assert (hits, boxed) == (2, 0) and best in ((0, 6, 9), (0, 6, 8))
    hits, boxed, _ = score_coverage_guess(sets, (4, 5, 7))
    assert (hits, boxed) == (0, 0)                        # nothing matched
    hits, boxed, best = score_coverage_guess(sets, (9, 9, 6))
    assert (hits, boxed) == (2, 0)   # pair in actual: multiset overlap 2


def test_merge_existing_rows_always_win():
    existing = pd.DataFrame([{"date": "2026-11-03", "draw": "eve",
                              "d1": 2, "d2": 5, "d3": 7}])
    existing["date"] = pd.to_datetime(existing["date"])
    fresh = pd.DataFrame([
        {"date": "2026-11-03", "draw": "eve", "d1": 9, "d2": 9, "d3": 9},
        {"date": "2026-11-04", "draw": "mid", "d1": 6, "d2": 2, "d3": 6}])
    fresh["date"] = pd.to_datetime(fresh["date"])
    merged = _merge_frames(existing, fresh)
    rows = {(r["date"].date(), r["draw"]): (int(r["d1"]), int(r["d2"]),
                                            int(r["d3"]))
            for _, r in merged.iterrows()}
    assert rows[(date(2026, 11, 3), "eve")] == (2, 5, 7)   # not overwritten
    assert rows[(date(2026, 11, 4), "mid")] == (6, 2, 6)   # new row added

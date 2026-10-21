"""Lotto-Muse v2 tests: feature causality (no look-ahead), lane
determinism, and scoring sanity. Run: venv/bin/python -m pytest tests_v2.py
"""
import numpy as np
import pandas as pd

from lotto_muse import (featurize, lane_due, lane_hotdigits, lane_random,
                        load_history, score_order_free, target_of)


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

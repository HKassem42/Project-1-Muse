"""Lotto-Muse v2: Michigan Daily 3 experimental prediction engine.

Fun research project. Lottery draws are independent random events, so no
model is expected to beat chance -- the backtest and dashboard exist to
measure exactly that, now over 5,700+ verified real draws (2019->today,
cross-validated backfill, see backfill.py).

Scoring is ORDER-FREE (per Kass): only whether the 3 digits are right,
regardless of order.
  - box_hit:      sorted(guess) == sorted(actual)  (all 3 digits right)
  - digits_right: multiset intersection size (0..3)

v2 changes (2026-10-21):
  - History is the full verified backfill (was: last ~35 days only).
  - Features v2 (71): last-5 digits, digit freq over 20/60/180 draws,
    days-since-each-digit, rolling sum/position stats, parity streak,
    day-of-week, evening flag. Every feature uses ONLY past draws
    (shift-based; truncation-tested in tests_v2.py).
  - Models train on the last 1,000 draws (was: 30).
  - Ensemble adds a Gradient Boosting voter next to hot / RF / LSTM.
  - Strategy LANES: every draw also gets independent guesses from
    hot / hot-digits / due-digits / pattern-sum strategies plus a seeded
    RANDOM CONTROL lane, each scored on its own scoreboard (logs/lanes.csv).
    A lane only means something if it beats the control lane.

Usage:
  python lotto_muse.py              daily run
  python lotto_muse.py --backtest   walk-forward backtest (v2)
  python lotto_muse.py --backtest --features v1   original recipe, for A/B
"""
import argparse
import hashlib
import pickle
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from scraper import scrape_daily3

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "history.csv"
LOG_FILE = BASE / "logs" / "guesses.csv"
LANES_FILE = BASE / "logs" / "lanes.csv"
MODEL_DIR = BASE / "models"
TRAIN_N = 1000       # draws used to train the ML models
MIN_PAST = 180       # history needed before a sample's features are full
LANES = ("ensemble", "hot", "hotdigits", "due", "pattern", "random")

for d in ("data", "logs", "models"):
    (BASE / d).mkdir(exist_ok=True)


# ---------------------------------------------------------------- history
def load_history() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return (df.assign(_o=df["draw"].map({"mid": 0, "eve": 1}))
                  .sort_values(["date", "_o"]).drop(columns="_o")
                  .reset_index(drop=True))
    return pd.DataFrame(columns=["date", "draw", "d1", "d2", "d3"])


def update_history() -> pd.DataFrame:
    """Scrape latest results and merge into history.csv. Fails loudly."""
    print("[1/6] Updating history...")
    existing = load_history()
    fresh = scrape_daily3()
    if fresh is None or len(fresh) == 0:
        if len(existing) == 0:
            raise SystemExit(
                "ERROR: scraper returned no data and no history exists. "
                "Refusing to train on fake data.")
        print(f"  scraper returned nothing new; keeping {len(existing)} draws")
        return existing
    combined = (pd.concat([existing, fresh])
                  .assign(_o=lambda df: df["draw"].map({"mid": 0, "eve": 1}))
                  .drop_duplicates(subset=["date", "draw"], keep="last")
                  .sort_values(["date", "_o"]).drop(columns="_o")
                  .reset_index(drop=True))
    combined.to_csv(DATA_FILE, index=False)
    print(f"  history: {len(combined)} draws "
          f"({combined['date'].min().date()} -> {combined['date'].max().date()})")
    return combined


def draws_list(df: pd.DataFrame):
    """Chronological list of (date, draw, d1, d2, d3)."""
    return [(r["date"], r["draw"], int(r["d1"]), int(r["d2"]), int(r["d3"]))
            for _, r in df.iterrows()]


# ---------------------------------------------------------------- features
def featurize(past, target_date, target_is_eve):
    """Features v2 (71) from past draws only. past: list of (d1,d2,d3),
    most-recent last. Nothing from the target draw is used."""
    past = [tuple(d) for d in past]
    last5 = [d for draw in past[-5:] for d in draw]                # 15
    while len(last5) < 15:
        last5 = [0] * (15 - len(last5)) + last5
    feats = list(last5)
    flat_all = [d for draw in past for d in draw]
    for w in (20, 60, 180):                                        # 30
        window = flat_all[-3 * w:]
        n = max(len(window), 1)
        feats += [window.count(x) / n for x in range(10)]
    for x in range(10):                                            # 10
        since = 90
        for i, draw in enumerate(reversed(past)):
            if x in draw:
                since = i
                break
        feats.append(min(since, 90) / 90.0)
    sums = np.array([sum(d) for d in past], dtype=float)
    for w in (14, 60):                                             # 4
        s = sums[-w:]
        feats += [s.mean() / 27.0, min(s.std(), 10.0) / 10.0]
    arr = np.array(past, dtype=float)
    for w in (14, 60):                                             # 6
        feats += list(arr[-w:].mean(axis=0) / 9.0)
    last = past[-1]                                                # 3
    feats += [sum(last) / 27.0,
              sum(1 for d in last if d % 2 == 0) / 3.0,
              sum(1 for d in last if d >= 5) / 3.0]
    streak, sign = 0, 0                                            # 1
    for d in reversed(past):
        s = 1 if sum(d) >= 14 else -1
        if sign == 0:
            sign = s
        if s != sign:
            break
        streak += 1
    feats.append(sign * min(streak, 20) / 20.0)
    feats += [pd.Timestamp(target_date).dayofweek / 6.0,           # 2
              1.0 if target_is_eve else 0.0]
    return np.array(feats, dtype=float)


def featurize_v1(past, target_date, target_is_eve):
    """Original 30-feature recipe (kept for A/B backtests)."""
    digits = [d for draw in past[-5:] for d in draw]
    while len(digits) < 15:
        digits = [0] * (15 - len(digits)) + digits
    window = past[-20:]
    flat = [d for draw in window for d in draw]
    freqs = [flat.count(x) for x in range(10)]
    last = past[-1]
    return np.array(digits + freqs +
                    [sum(last), sum(1 for d in last if d % 2 == 0),
                     sum(1 for d in last if d >= 5),
                     pd.Timestamp(target_date).dayofweek,
                     1 if target_is_eve else 0], dtype=float)


def target_of(draw):
    """Order-free target: sorted digit triple."""
    _, _, d1, d2, d3 = draw
    return tuple(sorted((d1, d2, d3)))


# ---------------------------------------------------------------- scoring
def score_order_free(guess, actual):
    """guess, actual: sorted triples. Returns (box_hit, digits_right)."""
    box = 1 if tuple(guess) == tuple(actual) else 0
    g, a = list(guess), list(actual)
    hits = 0
    for d in g:
        if d in a:
            hits += 1
            a.remove(d)
    return box, hits


# ---------------------------------------------------------------- lanes
def lane_hot(past_draws):
    triples = [target_of(d) for d in past_draws[-100:]]
    return max(set(triples), key=triples.count)


def lane_hotdigits(past_draws):
    flat = [d for draw in past_draws[-180:] for d in draw]
    top = sorted(range(10), key=lambda x: -flat.count(x))[:3]
    return tuple(sorted(top))


def lane_due(past_draws):
    since = {}
    for x in range(10):
        s = 10 ** 6
        for i, d in enumerate(reversed(past_draws)):
            if x in d[2:]:
                s = i
                break
        since[x] = s
    top = sorted(range(10), key=lambda x: (-since[x], x))[:3]
    return tuple(sorted(top))


def lane_pattern(past_draws):
    sums = [sum(d[2:]) for d in past_draws[-180:]]
    best_sum = max(set(sums), key=sums.count)
    cands = [target_of(d) for d in past_draws[-300:] if sum(d[2:]) == best_sum]
    if not cands:
        return lane_hot(past_draws)
    return max(set(cands), key=cands.count)


def lane_random(date, draw):
    seed = int(hashlib.sha256(f"{date}-{draw}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return tuple(sorted(rng.randint(0, 9) for _ in range(3)))


def lane_guesses(past_draws, date, draw):
    return {"hot": lane_hot(past_draws),
            "hotdigits": lane_hotdigits(past_draws),
            "due": lane_due(past_draws),
            "pattern": lane_pattern(past_draws),
            "random": lane_random(date, draw)}


# ---------------------------------------------------------------- models
class TinyLSTM:
    """Minimal LSTM sequence voter (last-10-draw sequences)."""

    def __init__(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(42)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=3, hidden_size=32,
                                    num_layers=1, batch_first=True)
                self.drop = nn.Dropout(0.3)
                self.fc = nn.Linear(32, 30)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.drop(out[:, -1, :])
                return self.fc(out).view(-1, 3, 10)

        self.torch = torch
        self.net = Net()

    def fit(self, seqs, targets, epochs=30):
        torch, nn = self.torch, self.torch.nn
        X = torch.tensor(np.array(seqs), dtype=torch.float32) / 9.0
        y = torch.tensor(np.array(targets), dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            p = self.net(X)
            loss = sum(loss_fn(p[:, i, :], y[:, i]) for i in range(3))
            loss.backward()
            opt.step()
        self.final_loss = loss.item()

    def predict(self, seq):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.float32) / 9.0
            p = self.net(x)
            return tuple(int(torch.argmax(p[0, i]).item()) for i in range(3))


def train_models(draws, features="v2", train_n=TRAIN_N):
    """Train on the most recent train_n draws. Returns dict of models."""
    feat = featurize if features == "v2" else featurize_v1
    min_past = MIN_PAST if features == "v2" else 20
    draws = draws[-max(train_n, min_past + 50):]
    X, Y = [], []
    for i in range(min_past, len(draws)):
        date, draw, *_ = draws[i]
        X.append(feat([d[2:] for d in draws[:i]], date, draw == "eve"))
        Y.append(target_of(draws[i]))
    X, Y = np.array(X), np.array(Y)

    models = {}
    triples = [tuple(t) for t in Y]
    models["hot"] = max(set(triples), key=triples.count)

    models["rf"], models["gb"] = [], []
    for pos in range(3):
        rf = RandomForestClassifier(n_estimators=150, max_depth=10,
                                    random_state=42, n_jobs=-1)
        rf.fit(X, Y[:, pos])
        models["rf"].append(rf)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                            max_depth=3, random_state=42)
        gb.fit(X, Y[:, pos])
        models["gb"].append(gb)

    seq_len = 10
    seqs = [[list(d[2:]) for d in draws[i - seq_len:i]]
            for i in range(min_past, len(draws))]
    lstm = TinyLSTM()
    lstm.fit(seqs, [tuple(t) for t in Y])
    models["lstm"] = lstm
    return models


def predict_next(models, past_draws, target_date, target_is_eve,
                 features="v2"):
    """Ensemble vote of hot / RF / GB / LSTM. Returns sorted triple."""
    feat = featurize if features == "v2" else featurize_v1
    past_digits = [d[2:] for d in past_draws]
    x = feat(past_digits, target_date, target_is_eve).reshape(1, -1)
    votes = {"hot": tuple(models["hot"]),
             "rf": tuple(int(models["rf"][p].predict(x)[0]) for p in range(3)),
             "gb": tuple(int(models["gb"][p].predict(x)[0]) for p in range(3)),
             "lstm": models["lstm"].predict([list(d) for d in past_digits[-10:]])}
    counts = {}
    for v in votes.values():
        counts[v] = counts.get(v, 0) + 1
    # majority vote; tie-break toward RF, then GB, then LSTM, then hot
    order = [votes["rf"], votes["gb"], votes["lstm"], votes["hot"]]
    best = max(counts.items(),
               key=lambda kv: (kv[1], -order.index(kv[0]) if kv[0] in order else 9))
    return best[0], votes


# ---------------------------------------------------------------- backtest
def backtest(df, features="v2", n_test=600, retrain_every=50):
    print(f"=== WALK-FORWARD BACKTEST ({features}, last {n_test} draws, "
          f"order-free) ===")
    draws = draws_list(df)
    start = len(draws) - n_test
    lanes = {name: [0, 0] for name in LANES}   # [box, digits]
    rng = random.Random(1234)
    rand = [0, 0]
    models, trained_at = None, -10 ** 9
    train_n = TRAIN_N if features == "v2" else 30
    for idx in range(start, len(draws)):
        if models is None or idx - trained_at >= retrain_every:
            window = draws[max(0, idx - train_n):idx] if features == "v2" \
                else draws[idx - 30:idx]
            models = train_models(window, features=features,
                                  train_n=train_n)
            trained_at = idx
        date, draw, *_ = draws[idx]
        past = draws[:idx]
        guess, _ = predict_next(models, past, date, draw == "eve",
                                features=features)
        actual = target_of(draws[idx])
        all_guesses = {"ensemble": guess, **lane_guesses(past, date, draw)}
        for name, g in all_guesses.items():
            box, hits = score_order_free(g, actual)
            lanes[name][0] += box
            lanes[name][1] += hits
        rguess = tuple(sorted(rng.randint(0, 9) for _ in range(3)))
        box, hits = score_order_free(rguess, actual)
        rand[0] += box
        rand[1] += hits
        if (idx - start + 1) % 100 == 0:
            print(f"  ... {idx - start + 1}/{n_test} draws scored")
    print(f"\n{'lane':>10} | box hits | digits/draw (chance ~0.71-0.90)")
    for name in LANES:
        box, hits = lanes[name]
        print(f"{name:>10} | {box:>4}/{n_test} = {box / n_test:6.3%} | "
              f"{hits / n_test:.3f}")
    print(f"{'pure-rng':>10} | {rand[0]:>4}/{n_test} = {rand[0] / n_test:6.3%} | "
          f"{rand[1] / n_test:.3f}")
    print("\nChance box-hit rate ~ 1/220 = 0.455% (uniform draws). Expected "
          "digits/draw for the seeded 'random' LANE differs from pure-rng "
          "only by seed luck — both are controls.")


# ---------------------------------------------------------------- daily run
def next_draws(df):
    last = df.iloc[-1]
    last_date = pd.Timestamp(last["date"]).date()
    out = []
    if last["draw"] == "mid":
        out += [(last_date, "eve"), (last_date + timedelta(days=1), "mid")]
    else:
        out += [(last_date + timedelta(days=1), "mid"),
                (last_date + timedelta(days=1), "eve")]
    return out


def score_pending(df):
    """Score logged ensemble guesses and lane guesses whose actuals exist."""
    hist = {(pd.Timestamp(r["date"]).date(), r["draw"]): target_of(
        (r["date"], r["draw"], int(r["d1"]), int(r["d2"]), int(r["d3"])))
        for _, r in df.iterrows()}
    if LOG_FILE.exists():
        logs = pd.read_csv(LOG_FILE, dtype=str)
        changed = False
        for idx, row in logs.iterrows():
            if pd.notna(row.get("box_hit")) and str(row.get("box_hit")) != "":
                continue
            key = (pd.Timestamp(row["date"]).date(), row["draw"])
            if key in hist:
                guess = tuple(int(row[f"g{i}"]) for i in (1, 2, 3))
                box, hits = score_order_free(guess, hist[key])
                logs.loc[idx, "box_hit"] = str(box)
                logs.loc[idx, "digits_right"] = str(hits)
                changed = True
                print(f"  scored {key[0]} {key[1]}: guess "
                      f"{''.join(map(str, guess))} -> box={box} digits={hits}/3")
        if changed:
            logs.to_csv(LOG_FILE, index=False)
    if LANES_FILE.exists():
        lanes = pd.read_csv(LANES_FILE, dtype=str)
        changed = False
        for idx, row in lanes.iterrows():
            if pd.notna(row.get("box_hit")) and str(row.get("box_hit")) != "":
                continue
            key = (pd.Timestamp(row["date"]).date(), row["draw"])
            if key in hist:
                guess = tuple(int(c) for c in row["guess"].zfill(3))
                box, hits = score_order_free(guess, hist[key])
                lanes.loc[idx, "box_hit"] = str(box)
                lanes.loc[idx, "digits_right"] = str(hits)
                changed = True
        if changed:
            lanes.to_csv(LANES_FILE, index=False)
    return hist


def daily_run():
    df = update_history()
    draws = draws_list(df)
    print("[2/6] Scoring previous guesses + lanes...")
    score_pending(df)
    print(f"[3/6] Training on last {TRAIN_N} draws (of {len(draws)})...")
    models = train_models(draws)
    print(f"  hot baseline: {''.join(map(str, models['hot']))} | "
          f"lstm final loss: {models['lstm'].final_loss:.3f}")
    print("[4/6] Predicting next draws (ensemble + lanes)...")
    new_rows, lane_rows = [], []
    for date, draw in next_draws(df):
        guess, parts = predict_next(models, draws, pd.Timestamp(date),
                                    draw == "eve")
        new_rows.append({"date": str(date), "draw": draw,
                         "g1": guess[0], "g2": guess[1], "g3": guess[2],
                         "hot": "".join(map(str, parts["hot"])),
                         "rf": "".join(map(str, parts["rf"])),
                         "gb": "".join(map(str, parts["gb"])),
                         "lstm": "".join(map(str, parts["lstm"])),
                         "box_hit": "", "digits_right": ""})
        lg = {"ensemble": guess, **lane_guesses(draws, date, draw)}
        for name, g in lg.items():
            lane_rows.append({"date": str(date), "draw": draw, "lane": name,
                              "guess": "".join(map(str, g)),
                              "box_hit": "", "digits_right": ""})
        print(f"  {date} {draw}: ensemble {''.join(map(str, guess))} "
              f"(hot {''.join(map(str, parts['hot']))} / "
              f"rf {''.join(map(str, parts['rf']))} / "
              f"gb {''.join(map(str, parts['gb']))} / "
              f"lstm {''.join(map(str, parts['lstm']))})")
    print("[5/6] Logging...")
    new_df = pd.DataFrame(new_rows)
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE, dtype=str)
        if "gb" not in old.columns:
            old["gb"] = ""
        seen = set(zip(old["date"].astype(str), old["draw"].astype(str)))
        new_df = new_df[~new_df.apply(
            lambda r: (str(r["date"]), str(r["draw"])) in seen, axis=1)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(LOG_FILE, index=False)
    lane_df = pd.DataFrame(lane_rows)
    if LANES_FILE.exists():
        old = pd.read_csv(LANES_FILE, dtype=str)
        seen = set(zip(old["date"].astype(str), old["draw"].astype(str),
                       old["lane"].astype(str)))
        lane_df = lane_df[~lane_df.apply(
            lambda r: (str(r["date"]), str(r["draw"]), str(r["lane"])) in seen,
            axis=1)]
        lane_df = pd.concat([old, lane_df], ignore_index=True)
    lane_df.to_csv(LANES_FILE, index=False)
    print("[6/6] Scoreboards")
    logs = pd.read_csv(LOG_FILE, dtype=str)
    done = logs[pd.to_numeric(logs["box_hit"], errors="coerce").notna()]
    if len(done):
        b = pd.to_numeric(done["box_hit"]).sum()
        d = pd.to_numeric(done["digits_right"]).mean()
        print(f"  ensemble live: box hits {int(b)}/{len(done)} | "
              f"avg digits right {d:.2f}/3 over {len(done)} scored")
    if LANES_FILE.exists():
        lanes = pd.read_csv(LANES_FILE, dtype=str)
        scored = lanes[pd.to_numeric(lanes["box_hit"], errors="coerce").notna()]
        for name, grp in scored.groupby("lane"):
            bb = pd.to_numeric(grp["box_hit"]).sum()
            dd = pd.to_numeric(grp["digits_right"]).mean()
            print(f"  lane {name:>10}: box {int(bb)}/{len(grp)} | "
                  f"digits {dd:.2f}/3")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--features", default="v2", choices=["v1", "v2"])
    ap.add_argument("--n-test", type=int, default=600)
    args = ap.parse_args()
    if args.backtest:
        df = update_history()
        backtest(df, features=args.features, n_test=args.n_test)
    else:
        daily_run()
    print("\nDone.")

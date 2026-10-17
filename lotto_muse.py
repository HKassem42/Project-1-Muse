"""Lotto-Muse: Michigan Daily 3 experimental prediction engine.

Fun research project. Lottery draws are independent random events, so no
model is expected to beat chance -- the backtest and dashboard exist to
measure exactly that.

Scoring is ORDER-FREE (per Kass): only whether the 3 digits are right,
regardless of order.
  - box_hit:      sorted(guess) == sorted(actual)  (all 3 digits right)
  - digits_right: multiset intersection size (0..3)

Usage:
  python lotto_muse.py            daily run: update history, score old guesses,
                                  retrain on last 30 draws, predict next 2 draws
  python lotto_muse.py --backtest walk-forward backtest over full history
"""
import os
import sys
import pickle
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from scraper import scrape_daily3

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "history.csv"
LOG_FILE = BASE / "logs" / "guesses.csv"
MODEL_DIR = BASE / "models"
WINDOW = 30          # rolling training window (draws)
WARMUP = 40          # draws needed before first backtest prediction

for d in ("data", "logs", "models"):
    (BASE / d).mkdir(exist_ok=True)


# ---------------------------------------------------------------- history
def load_history() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "draw"]).reset_index(drop=True)
    return pd.DataFrame(columns=["date", "draw", "d1", "d2", "d3"])


def update_history() -> pd.DataFrame:
    """Scrape latest results and merge into history.csv. Fails loudly."""
    print("[1/5] Updating history...")
    existing = load_history()
    fresh = scrape_daily3()
    if fresh is None or len(fresh) == 0:
        if len(existing) == 0:
            raise SystemExit(
                "ERROR: scraper returned no data and no history exists. "
                "Refusing to train on fake data.")
        print(f"  scraper returned nothing new; keeping {len(existing)} draws")
        return existing
    if len(existing):
        combined = pd.concat([existing, fresh])
    else:
        combined = fresh
    combined = (combined.drop_duplicates(subset=["date", "draw"], keep="last")
                        .sort_values(["date", "draw"]).reset_index(drop=True))
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
    """Feature vector from past draws (list of (d1,d2,d3), most-recent last).

    30 features: last 5 draws' digits (15) + digit freqs over last 20 (10) +
    last-draw sum/even/high (3) + day of week (1) + is_evening (1).
    """
    digits = [d for draw in past[-5:] for d in draw]          # 15
    while len(digits) < 15:                                   # pad if short
        digits = [0] * (15 - len(digits)) + digits
    window = past[-20:]
    flat = [d for draw in window for d in draw]
    freqs = [flat.count(x) for x in range(10)]                # 10
    last = past[-1]
    s = sum(last)
    feats = (digits + freqs +
             [s, sum(1 for d in last if d % 2 == 0),
              sum(1 for d in last if d >= 5),
              pd.Timestamp(target_date).dayofweek,
              1 if target_is_eve else 0])
    return np.array(feats, dtype=float)


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


# ---------------------------------------------------------------- models
class TinyLSTM:
    """Minimal LSTM for the 'deep learning' slot. Small on purpose:
    30 training samples cannot support a big network."""

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

    def fit(self, seqs, targets):
        torch, nn = self.torch, self.torch.nn
        X = torch.tensor(np.array(seqs), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(40):
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
            p = self.net(torch.tensor([seq], dtype=torch.float32))
            return tuple(int(torch.argmax(p[0, i]).item()) for i in range(3))


def train_models(draws):
    """Train on a list of chronological draws. Returns dict of models."""
    X, Y = [], []
    for i in range(20, len(draws)):
        date, draw, *_ = draws[i]
        X.append(featurize([d[2:] for d in draws[:i]], date, draw == "eve"))
        Y.append(target_of(draws[i]))
    X = np.array(X)
    Y = np.array(Y)

    models = {}
    # 1) hot-multiset baseline: most common sorted triple in window
    triples = [tuple(t) for t in Y]
    models["hot"] = max(set(triples), key=triples.count)

    # 2) random forest: 3 position classifiers on sorted targets
    rfs = []
    for pos in range(3):
        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, n_jobs=-1)
        clf.fit(X, Y[:, pos])
        rfs.append(clf)
    models["rf"] = rfs

    # 3) tiny LSTM on last-5-draw digit sequences
    seqs = [[list(d[2:]) for d in draws[i - 5:i]] for i in range(20, len(draws))]
    lstm = TinyLSTM()
    lstm.fit(seqs, [tuple(t) for t in Y])
    models["lstm"] = lstm
    return models


def predict_next(models, past_draws, target_date, target_is_eve):
    """Ensemble vote of hot / RF / LSTM. Returns sorted triple."""
    past_digits = [d[2:] for d in past_draws]
    x = featurize(past_digits, target_date, target_is_eve).reshape(1, -1)
    votes = [models["hot"]]
    votes.append(tuple(int(models["rf"][p].predict(x)[0]) for p in range(3)))
    seq = [list(d) for d in past_digits[-5:]]
    votes.append(models["lstm"].predict(seq))
    # majority vote, tie-break toward RF
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == votes[1]))
    return best[0], {"hot": votes[0], "rf": votes[1], "lstm": votes[2]}


# ---------------------------------------------------------------- backtest
def backtest(df):
    print("=== WALK-FORWARD BACKTEST (order-free scoring) ===")
    draws = draws_list(df)
    rng = random.Random(1234)
    stats = {"model": [0, 0, 0], "random": [0, 0, 0]}  # [box, digits, n]
    n = 0
    for i in range(WARMUP, len(draws)):
        window = draws[i - WINDOW:i]
        models = train_models(window)
        date, draw, *_ = draws[i]
        guess, _ = predict_next(models, window, date, draw == "eve")
        actual = target_of(draws[i])
        box, hits = score_order_free(guess, actual)
        stats["model"][0] += box
        stats["model"][1] += hits
        rguess = tuple(sorted(rng.randint(0, 9) for _ in range(3)))
        rbox, rhits = score_order_free(rguess, actual)
        stats["random"][0] += rbox
        stats["random"][1] += rhits
        n += 1
        if n % 10 == 0:
            print(f"  ... {n} draws scored")
    for name in ("model", "random"):
        box, hits, _ = stats[name][0], stats[name][1], n
        print(f"{name:>6}: box-hit rate {box}/{n} = {box/n:.3%} | "
              f"avg digits right {hits/n:.2f}/3")
    print(f"\nChance baseline for box hits on uniform draws ~ "
          f"{1/220:.3%} per guess (220 possible multisets).")


# ---------------------------------------------------------------- daily run
def next_draws(df):
    """Next 2 upcoming draws after the latest actual."""
    last = df.iloc[-1]
    last_date = pd.Timestamp(last["date"]).date()
    out = []
    if last["draw"] == "mid":
        out += [(last_date, "eve"), (last_date + timedelta(days=1), "mid")]
    else:
        out += [(last_date + timedelta(days=1), "mid"),
                (last_date + timedelta(days=1), "eve")]
    return out


def score_pending_guesses(df):
    """Score logged guesses whose actuals are now known."""
    if not LOG_FILE.exists():
        return pd.DataFrame()
    logs = pd.read_csv(LOG_FILE)
    hist = {(pd.Timestamp(r["date"]).date(), r["draw"]): target_of(
        (r["date"], r["draw"], int(r["d1"]), int(r["d2"]), int(r["d3"])))
        for _, r in df.iterrows()}
    changed = False
    for idx, row in logs.iterrows():
        if pd.notna(row.get("box_hit")) and str(row.get("box_hit")) != "":
            continue
        key = (pd.Timestamp(row["date"]).date(), row["draw"])
        if key in hist:
            guess = tuple(int(row[f"g{i}"]) for i in (1, 2, 3))
            box, hits = score_order_free(guess, hist[key])
            logs.loc[idx, "box_hit"] = box
            logs.loc[idx, "digits_right"] = hits
            changed = True
            print(f"  scored {key[0]} {key[1]}: guess {''.join(map(str,guess))} "
                  f"-> box={box} digits={hits}/3")
    if changed:
        logs.to_csv(LOG_FILE, index=False)
    return logs


def daily_run():
    df = update_history()
    draws = draws_list(df)
    print("[2/5] Scoring previous guesses...")
    logs = score_pending_guesses(df)
    print("[3/5] Training on last 30 draws...")
    window = draws[-WINDOW:]
    models = train_models(window)
    print(f"  hot baseline: {''.join(map(str, models['hot']))} | "
          f"lstm final loss: {models['lstm'].final_loss:.3f}")
    print("[4/5] Predicting next draws...")
    new_rows = []
    for date, draw in next_draws(df):
        guess, parts = predict_next(models, draws, pd.Timestamp(date), draw == "eve")
        new_rows.append({"date": str(date), "draw": draw,
                         "g1": guess[0], "g2": guess[1], "g3": guess[2],
                         "hot": "".join(map(str, parts["hot"])),
                         "rf": "".join(map(str, parts["rf"])),
                         "lstm": "".join(map(str, parts["lstm"])),
                         "box_hit": "", "digits_right": ""})
        print(f"  {date} {draw}: {''.join(map(str, guess))} "
              f"(hot {''.join(map(str,parts['hot']))} / "
              f"rf {''.join(map(str,parts['rf']))} / "
              f"lstm {''.join(map(str,parts['lstm']))})")
    print("[5/5] Logging...")
    new_df = pd.DataFrame(new_rows)
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE, dtype=str)
        old = old[~old.set_index(["date", "draw"]).index.isin(
            new_df.set_index(["date", "draw"]).index)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(LOG_FILE, index=False)
    # dashboard
    if len(logs):
        done = logs[pd.to_numeric(logs["box_hit"], errors="coerce").notna()]
        if len(done):
            b = pd.to_numeric(done["box_hit"]).sum()
            d = pd.to_numeric(done["digits_right"]).mean()
            print(f"\n=== LOTTO-MUSE DASHBOARD ({len(done)} scored) ===")
            print(f"box hits: {int(b)}/{len(done)} | avg digits right: {d:.2f}/3")


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        df = update_history()
        backtest(df)
    else:
        daily_run()
    print("\nDone.")

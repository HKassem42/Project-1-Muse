# Lotto-Muse

A fun Michigan Daily 3 prediction experiment. A Python engine scrapes results
daily, trains ML models on the last 1,000 of **5,700+ verified real draws
(2019 → today)**, predicts upcoming MID/EVE draws, and logs every guess.
Scoring is **order-free**: only whether the 3 digits are right, regardless
of order.

Lottery draws are independent random events — no model here is expected to
beat chance. The point of the project is the machinery and the honest
scoreboard: multiple strategies race a random control lane, and the
dashboard tracks whether anything beats chance over time.

## How it works

```
backfill (verified) → scrape → features → train → predict → log → score → retrain (daily)
```

- **Verified backfill** (`backfill.py`) — builds the 2019→today history from
  three independent archives (LotteryCorner, BeatLottery, Lottery.net);
  a draw is imported only if at least two sources agree digit-for-digit.
  (BeatLottery's midday numbers are wrong in stretches — the cross-check
  catches it. An earlier side project trained on *randomly generated*
  "history" from a scraper fallback; this project never does that.)
- **Data collector** (`scraper.py`) — pulls the newest Michigan Daily 3
  MID/EVE results from a public results archive, plus a **fast layer**
  (v3, 2026-11-04): three independent latest-results pages, accepted
  only when at least 2 of 3 agree digit-for-digit. The afternoon run
  waits for the same-day midday result (polling the fast layer) so the
  evening pick trains on it. Recorded history can never be overwritten
  by a scrape — conflicts are reported, existing rows win.
- **Brain** (`lotto_muse.py`) — 71 leak-free features (digit frequencies
  over 20/60/180 draws, days-since-each-digit, rolling stats, streaks;
  truncation-tested in `tests_v2.py`) and four voters:
  1. hot-multiset baseline
  2. random forest (150 trees)
  3. gradient boosting
  4. tiny LSTM (32 hidden units, 10-draw sequences)
- **Guess construction (v3)** — the official pick is no longer a majority
  vote over whole triples. Per-digit scores are blended from the RF/GB
  per-position probabilities (expected digit counts), new per-digit
  binary classifiers ("does digit *d* appear next?"), 180-draw inclusion
  frequencies, and voter support; the pick is the **3-distinct-digit**
  set with the highest total score (Kass's 2-1 parity preference survives
  as a tie-breaker bonus). Distinct sets are the provably best pattern
  class under order-free scoring: 0.60% box rate and 0.813 expected
  digits/draw, vs 0.30% / 0.570 for a pair guess (brute-forced over all
  1,000 outcomes).
- **Strategy lanes** — independent guesses per draw from hot / hot-digits /
  due-digits / pattern-sum strategies plus a seeded **random control**,
  each scored on its own scoreboard (`logs/lanes.csv`).
- **Logger** — predicts the next draws, saves guesses to `logs/guesses.csv`,
  and auto-scores them when actuals arrive (`box_hit`, `digits_right`).

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python backfill.py --years 2019-2026   # one-time verified history build
python lotto_muse.py                   # morning run (midday pick)
python lotto_muse.py --evening         # afternoon run (evening pick, waits for midday)
python lotto_muse.py --backtest        # walk-forward backtest (v3 ensemble + v2-vote A/B)
python lotto_muse.py --backtest --features v1   # original recipe, for A/B
python sync_dashboard.py               # sync CSVs to the Lovable dashboard
```

## Results so far

Walk-forward backtest over the last 600 real draws (order-free) —
v2-era table; the v3 ensemble went live 2026-11-05 and its backtest
A/B (v3 selector vs v2 vote, printed by `--backtest`) supersedes the
ensemble row as it accumulates:

| lane | box hits | digits/draw |
|---|---|---|
| v2 ensemble | 3/600 (chance ≈ 2.7) | **0.812** |
| v1 ensemble (original) | 7/600 | 0.758 |
| hot-digits lane | 4/600 | 0.807 |
| random controls | 4–7/600 | 0.723–0.740 |

Box hits are at chance for everything, exactly as expected; the v2 ensemble
leads on digits-per-draw, the "play the most probable combinations" effect.
The experiment continues daily via automation, with results on the dashboard.

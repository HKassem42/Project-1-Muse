# Lotto-Muse

A fun Michigan Daily 3 prediction experiment. A Python engine scrapes results
daily, trains small ML models on a rolling 30-draw window, predicts upcoming
MID/EVE draws, and logs every guess. Scoring is **order-free**: only whether
the 3 digits are right, regardless of order.

Lottery draws are independent random events — no model here is expected to
beat chance. The point of the project is the machinery and the honest
scoreboard: the dashboard tracks whether the models beat chance over time.

## How it works

```
scrape → features → train → predict → log → score → retrain (daily)
```

- **Data collector** (`scraper.py`) — pulls Michigan Daily 3 MID/EVE history
  from a public results archive (cross-validated against two independent sources).
- **Brain** (`lotto_muse.py`) — three models vote on each draw:
  1. hot-multiset baseline (most common digit triple in the window)
  2. random forest (100 trees, order-free sorted-digit targets)
  3. tiny LSTM (32 hidden units — small on purpose for 30 training samples)
- **Logger** — predicts the next draws, saves guesses to `logs/guesses.csv`,
  and auto-scores them when actuals arrive (`box_hit`, `digits_right`).
- **Retrainer** — every run trains fresh on the last 30 draws.

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python lotto_muse.py            # daily run: update, score, retrain, predict
python lotto_muse.py --backtest # walk-forward backtest over full history
python sync_dashboard.py        # sync CSVs to the Lovable dashboard database
```

## Results so far

Walk-forward backtest (21 draws): model 0% box hits, 0.95 digits right per
guess vs 0.71 for a random guesser — no edge, exactly as expected. The
experiment continues daily via automation, with results on the dashboard.

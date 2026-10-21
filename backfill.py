"""Lotto-Muse verified history backfill.

Downloads Michigan Daily 3 results from THREE independent archives and
imports only the draws at least two sources agree on digit-for-digit:

  A) LotteryCorner  lotterycorner.com/mi/daily-3-{midday,evening}/<year>
  B) BeatLottery    beatlottery.com/.../daily3/year/<year>
  C) Lottery.net    lottery.net/michigan/daily-3-{midday,evening}/numbers/<year>

(BeatLottery's midday numbers are wrong in long stretches — it was caught
by this cross-check on 2026-10-21 and loses every majority vote there.
LotteryCorner matched every independently verified anchor draw.)

Draws with no 2-source majority go to data/backfill/quarantine.csv with
all three values recorded — never into history. Existing rows in
data/history.csv are checked against the backfill where they overlap;
a conflict aborts the merge.

Sanity gates on the merged result: draws exist on Sundays (Michigan Daily
3 draws 7 days/week) and gaps in the calendar are reported.

Usage:
  python backfill.py --years 2020-2026            # build + merge
  python backfill.py --years 2020-2026 --dry-run  # build + report only
"""
import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "history.csv"
WORK = BASE / "data" / "backfill"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

LC_URL = "https://lotterycorner.com/mi/daily-3-{draw}/{year}"
BL_URL = ("https://www.beatlottery.com/lotteries/draws/country/us/state/mi"
          "/lottery/daily3/year/{year}")
LN_URL = "https://www.lottery.net/michigan/daily-3-{draw}/numbers/{year}"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
MON3 = {m[:3]: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"])}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def fetch(url: str, cache: Path) -> str:
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text()
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    time.sleep(0.5)
    return r.text


# ---------------------------------------------------------------- sources
def parse_lotterycorner(html: str, draw: str, year: int) -> dict:
    """{(date, draw): (d1, d2, d3)} from a LotteryCorner yearly page."""
    soup = BeautifulSoup(html, "lxml")
    out = {}
    pat = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|"
        r"October|November|December) (\d{1,2}), (\d{4})$")
    for tr in soup.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue
        m = pat.match(tds[0])
        if not m or int(m.group(3)) != year:
            continue
        if not re.fullmatch(r"\d{3}", tds[1]):
            continue
        d = date(year, MONTHS[m.group(1)], int(m.group(2)))
        out[(d, draw)] = tuple(int(c) for c in tds[1])
    return out


def parse_beatlottery(html: str, year: int) -> dict:
    """{(date, draw): (d1, d2, d3)} from a BeatLottery yearly page.

    Layout: a row 'Dec 31, 2025 12/31/2025 | Wed | Midday: 05 08 03' is
    followed by a dateless row holding the Evening numbers.
    """
    soup = BeautifulSoup(html, "lxml")
    out = {}
    tables = soup.find_all("table")
    if not tables:
        return out
    cur = None
    date_pat = re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2}), (\d{4})")
    num_pat = re.compile(r"^(Midday|Evening): (\d+) (\d+) (\d+)$")
    for tr in tables[0].find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        dm = date_pat.match(tds[0])
        if dm and int(dm.group(3)) == year:
            cur = date(year, MON3[dm.group(1)], int(dm.group(2)))
        for cell in tds:
            nm = num_pat.match(cell)
            if nm and cur is not None:
                draw = "mid" if nm.group(1) == "Midday" else "eve"
                out[(cur, draw)] = tuple(int(nm.group(i)) for i in (2, 3, 4))
    return out


def parse_lotterynet(html: str, draw: str, year: int) -> dict:
    """{(date, draw): (d1, d2, d3)} from a Lottery.net yearly page.

    Rows look like 'Wednesday December 31, 2025 | 5 8 3 | ...'. The weekday
    name must match the real calendar — a cheap parser sanity check.
    """
    soup = BeautifulSoup(html, "lxml")
    out = {}
    pat = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) "
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December) (\d{1,2}), (\d{4})$")
    for tr in soup.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue
        m = pat.match(tds[0])
        if not m or int(m.group(4)) != year:
            continue
        d = date(year, MONTHS[m.group(2)], int(m.group(3)))
        if WEEKDAYS[d.weekday()] != m.group(1):
            continue  # misaligned row — skip rather than trust
        digits = tds[1].split()
        if len(digits) != 3 or not all(x.isdigit() for x in digits):
            continue
        out[(d, draw)] = tuple(int(x) for x in digits)
    return out


# ---------------------------------------------------------------- build
def build_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (verified, quarantine) frames for one year (2-of-3 majority)."""
    src = {}
    a = {}
    for draw in ("midday", "evening"):
        html = fetch(LC_URL.format(draw=draw, year=year),
                     WORK / f"lc-{draw}-{year}.html")
        a.update(parse_lotterycorner(
            html, "mid" if draw == "midday" else "eve", year))
    src["lotterycorner"] = a
    src["beatlottery"] = parse_beatlottery(
        fetch(BL_URL.format(year=year), WORK / f"bl-{year}.html"), year)
    c = {}
    for draw in ("midday", "evening"):
        html = fetch(LN_URL.format(draw=draw, year=year),
                     WORK / f"ln-{draw}-{year}.html")
        c.update(parse_lotterynet(
            html, "mid" if draw == "midday" else "eve", year))
    src["lotterynet"] = c

    verified, quarantine = [], []
    keys = sorted(set().union(*[set(s) for s in src.values()]))
    for key in keys:
        d, draw = key
        vals = {name: s.get(key) for name, s in src.items()}
        present = [v for v in vals.values() if v is not None]
        winner = None
        for v in set(present):
            if present.count(v) >= 2:
                winner = v
                break
        if winner is not None:
            verified.append({"date": d, "draw": draw, "d1": winner[0],
                             "d2": winner[1], "d3": winner[2]})
        else:
            quarantine.append({
                "date": d, "draw": draw,
                **{name: "".join(map(str, v)) if v else ""
                   for name, v in vals.items()}})
    print(f"  {year}: LC={len(src['lotterycorner'])} "
          f"BL={len(src['beatlottery'])} LN={len(src['lotterynet'])} "
          f"verified={len(verified)} quarantine={len(quarantine)}")
    return pd.DataFrame(verified), pd.DataFrame(quarantine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2026",
                    help="e.g. 2020-2026 or 2023,2024")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if "-" in args.years:
        lo, hi = args.years.split("-")
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    print(f"Building cross-validated backfill for {years} ...")
    frames, quar = [], []
    for y in years:
        v, q = build_year(y)
        frames.append(v)
        quar.append(q)
    backfill = pd.concat(frames, ignore_index=True)
    quar = pd.concat(quar, ignore_index=True) if quar else pd.DataFrame()
    if not quar.empty:
        quar.to_csv(WORK / "quarantine.csv", index=False)
        print(f"  quarantine written: {len(quar)} draws NOT imported")

    backfill["date"] = pd.to_datetime(backfill["date"])
    print(f"Backfill total: {len(backfill)} draws "
          f"({backfill['date'].min().date()} -> {backfill['date'].max().date()})")

    sundays = backfill[backfill["date"].dt.dayofweek == 6]
    print(f"  Sunday draws: {len(sundays)} (must be > 0)")
    if len(sundays) == 0:
        raise SystemExit("ABORT: no Sunday draws — data does not look like "
                         "real Michigan Daily 3.")

    existing = pd.read_csv(DATA_FILE)
    existing["date"] = pd.to_datetime(existing["date"])
    key = ["date", "draw"]
    both = pd.merge(existing, backfill, on=key, suffixes=("_old", "_new"))
    conflicts = both[(both["d1_old"] != both["d1_new"]) |
                     (both["d2_old"] != both["d2_new"]) |
                     (both["d3_old"] != both["d3_new"])]
    if len(conflicts):
        print(conflicts.to_string())
        raise SystemExit(f"ABORT: {len(conflicts)} conflicts with existing "
                         "history — investigate before merging.")
    full = (pd.concat([existing, backfill])
              .drop_duplicates(subset=key, keep="last")
              .assign(_o=lambda df: df["draw"].map({"mid": 0, "eve": 1}))
              .sort_values(["date", "_o"]).drop(columns="_o")
              .reset_index(drop=True))
    print(f"  overlap with existing history: {len(both)} draws, all agree")
    print(f"  merged history: {len(full)} draws "
          f"({full['date'].min().date()} -> {full['date'].max().date()})")

    days = pd.date_range(full["date"].min(), full["date"].max(), freq="D")
    have = set(zip(full["date"], full["draw"]))
    gaps = [(d.date(), dr) for d in days for dr in ("mid", "eve")
            if (d, dr) not in have]
    print(f"  missing draws in range: {len(gaps)}"
          + (f" (first: {gaps[:5]})" if gaps else ""))

    if args.dry_run:
        print("Dry run — history.csv untouched.")
        return
    backup = DATA_FILE.with_suffix(".pre-backfill.csv")
    existing.to_csv(backup, index=False)
    full.to_csv(DATA_FILE, index=False)
    print(f"  backup of old history: {backup.name}")
    print(f"  wrote {DATA_FILE} ({len(full)} draws)")


if __name__ == "__main__":
    main()

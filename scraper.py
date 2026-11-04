"""Lotto-Muse data ingestion.

Two layers:
  1. Archive scrape: lotteryvalley.com's past-results archive
     (server-rendered tables, ~30 days). Cross-validated 2026-10-16
     against lotterydailypicks.com and duedigits.com.
  2. FAST latest-results layer (added 2026-11-04 to fix the afternoon
     lag): three independent "latest numbers" pages that post within
     minutes of a draw — Lottery Post, Michigan Lottery Numbers, and
     Lottery Valley's Daily 3 game page. A fast draw is accepted ONLY
     when at least 2 of the 3 sources agree digit-for-digit, the same
     2-of-3 standard the verified backfill uses. The archive page alone
     was still missing the same-day midday result at the 1:35 PM
     afternoon run on most days (Oct 30 / 31, Nov 1 / 3 / 4), which
     forced the evening pick to train on stale data.
"""
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, datetime

ARCHIVE_URL = "https://www.lotteryvalley.com/michigan/past-results"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

DATE_FMTS = ("%a, %b %d %Y", "%a, %b %d, %Y")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
MON3 = {m[:3]: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"])}

FAST_SOURCES = (
    ("lotterypost", "https://www.lotterypost.com/results/mi"),
    ("milnumbers", "https://michiganlotterynumbers.com/daily-3"),
    ("valley_game", "https://www.lotteryvalley.com/michigan/games/daily-3"),
)


def parse_date(s: str):
    s = s.strip()
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def fetch_archive_html() -> str:
    r = requests.get(ARCHIVE_URL, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def _rows_from_trs(trs, game_names=("Daily 3 Midday", "Daily 3 Evening"),
                   draw_names=("Midday", "Evening")):
    """Extract (date, draw, d1, d2, d3) from <tr> elements.

    Two layouts exist on the page:
      * top archive table: game cell = "Daily 3 Midday"/"Daily 3 Evening"
      * per-game table:    game cell = "Midday"/"Evening" (section is Daily 3)
    """
    rows = []
    for tr in trs:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        game = tds[1].get_text(strip=True)
        if game in game_names:
            draw = "mid" if "Midday" in game else "eve"
        elif game in draw_names:
            draw = "mid" if game == "Midday" else "eve"
        else:
            continue
        date = parse_date(tds[0].get_text(strip=True))
        if date is None:
            continue
        digits = [s.get_text(strip=True) for s in tds[2].find_all("span")]
        digits = [d for d in digits if re.fullmatch(r"\d", d)]
        if len(digits) != 3:
            continue
        d1, d2, d3 = (int(d) for d in digits)
        rows.append({"date": date, "draw": draw, "d1": d1, "d2": d2, "d3": d3})
    return rows


def parse_daily3(html: str) -> pd.DataFrame:
    """Extract Daily 3 rows. Returns date, draw, d1, d2, d3 sorted oldest->newest."""
    soup = BeautifulSoup(html, "lxml")
    # 1) per-game Daily 3 section (draw cell = Midday/Evening)
    start = html.find("Daily 3 Past Results")
    end = html.find("Daily 4 Past Results")
    rows = []
    if start != -1 and end != -1:
        section = BeautifulSoup(html[start:end], "lxml")
        rows += _rows_from_trs(section.find_all("tr"),
                               game_names=(), draw_names=("Midday", "Evening"))
    # 2) top archive table (game cell = "Daily 3 Midday"/"Daily 3 Evening")
    rows += _rows_from_trs(soup.find_all("tr"))
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "draw"])
    df["date"] = pd.to_datetime(df["date"])
    # chronological: midday draw comes before evening draw on a date
    return (df.assign(_o=df["draw"].map({"mid": 0, "eve": 1}))
              .sort_values(["date", "_o"]).drop(columns="_o")
              .reset_index(drop=True))


def scrape_daily3() -> pd.DataFrame:
    return parse_daily3(fetch_archive_html())


# ---------------------------------------------------------------- fast layer
def _page_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)


def parse_lotterypost_latest(text: str) -> dict:
    """{(date, draw): (d1,d2,d3)} from Lottery Post's Michigan page:
    'Daily 3 Midday Wednesday, November 4, 2026 Midday 6 2 6 ...'."""
    out = {}
    for draw, name in (("mid", "Midday"), ("eve", "Evening")):
        m = re.search(
            rf"Daily 3 {name}\s+\w+, (\w+) (\d+), (\d{{4}})\s+{name}\s+"
            r"(\d)\s+(\d)\s+(\d)", text)
        if m and m.group(1) in MONTHS:
            d = date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
            out[(d, draw)] = tuple(int(m.group(i)) for i in (4, 5, 6))
    return out


def parse_milnumbers_latest(text: str) -> dict:
    """{(date, draw): (d1,d2,d3)} from michiganlotterynumbers.com:
    'Wednesday, November 4, 2026 6 2 6 More Daily 3 Midday Numbers ...'."""
    out = {}
    pat = re.compile(
        r"\w+, (\w+) (\d+), (\d{4})\s+(\d)\s+(\d)\s+(\d)\s+"
        r"More Daily 3 (Midday|Evening) Numbers")
    for m in pat.finditer(text):
        if m.group(1) not in MONTHS:
            continue
        d = date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        out[(d, "mid" if m.group(7) == "Midday" else "eve")] = tuple(
            int(m.group(i)) for i in (4, 5, 6))
    return out


def parse_valley_latest(text: str) -> dict:
    """{(date, draw): (d1,d2,d3)} from Lottery Valley's Daily 3 page:
    'Daily 3 Midday Nov 4, 2026 $500 Top Prize 6 2 6 ...'."""
    out = {}
    pat = re.compile(
        r"Daily 3 (Midday|Evening)\s+(\w+) (\d+), (\d{4})\s+"
        r"\$500 Top Prize\s+(\d)\s+(\d)\s+(\d)")
    for m in pat.finditer(text):
        if m.group(2) not in MON3:
            continue
        d = date(int(m.group(4)), MON3[m.group(2)], int(m.group(3)))
        out[(d, "mid" if m.group(1) == "Midday" else "eve")] = tuple(
            int(m.group(i)) for i in (5, 6, 7))
    return out


_FAST_PARSERS = {"lotterypost": parse_lotterypost_latest,
                 "milnumbers": parse_milnumbers_latest,
                 "valley_game": parse_valley_latest}


def scrape_fast_latest() -> tuple[pd.DataFrame, dict]:
    """Latest draws from the 3 fast sources. A (date, draw) is returned
    only when >= 2 sources agree digit-for-digit. Returns (df, report)
    where report maps each seen key to {value: [sources]} plus errors."""
    seen, errors = {}, {}
    for name, url in FAST_SOURCES:
        try:
            for key, val in _FAST_PARSERS[name](_page_text(url)).items():
                seen.setdefault(key, {}).setdefault(val, []).append(name)
        except Exception as e:  # one dead source must not kill the layer
            errors[name] = f"{type(e).__name__}: {e}"
    rows = []
    for (d, draw), by_val in sorted(seen.items()):
        for val, sources in by_val.items():
            if len(sources) >= 2:
                rows.append({"date": d, "draw": draw, "d1": val[0],
                             "d2": val[1], "d3": val[2]})
                break
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df = (df.assign(_o=df["draw"].map({"mid": 0, "eve": 1}))
                .sort_values(["date", "_o"]).drop(columns="_o")
                .reset_index(drop=True))
    return df, {"seen": seen, "errors": errors}


if __name__ == "__main__":
    df = scrape_daily3()
    print(f"Got {len(df)} Daily 3 draws")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))

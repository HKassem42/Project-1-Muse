"""Lotto-Muse data ingestion.

Scrapes Michigan Daily 3 MID/EVE results from lotteryvalley.com's
past-results archive (server-rendered tables, updated after every draw).

Cross-validated 2026-10-16 against lotterydailypicks.com and duedigits.com:
all three sources agree on recent draws.
"""
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

ARCHIVE_URL = "https://www.lotteryvalley.com/michigan/past-results"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

DATE_FMTS = ("%a, %b %d %Y", "%a, %b %d, %Y")


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
    return df.sort_values(["date", "draw"]).reset_index(drop=True)


def scrape_daily3() -> pd.DataFrame:
    return parse_daily3(fetch_archive_html())


if __name__ == "__main__":
    df = scrape_daily3()
    print(f"Got {len(df)} Daily 3 draws")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))

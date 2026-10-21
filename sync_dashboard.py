"""Sync local Lotto-Muse CSVs -> Lovable dashboard DB (idempotent)."""
import json
import subprocess
import pandas as pd

PROJECT = "a1824c13-8733-445b-9f87-6d13929d7f94"
BASE = "/home/hatch/workspace/lotto-muse"


def q(sql):
    r = subprocess.run(
        ["/opt/hatch/bin/lovable", "call-tool", "--name", "query_database",
         "--arguments-json", json.dumps({"project_id": PROJECT, "sql": sql})],
        capture_output=True, text=True, timeout=60)
    d = json.loads(r.stdout)
    result = d.get("result", {}) or {}
    # NOTE: the CLI returns ok:true even when the DB call is rate-limited;
    # the real error lives in result.isError / content text.
    if not d.get("ok") or result.get("isError"):
        raise RuntimeError(str(result.get("content"))[:300])
    return result.get("structuredContent")


def esc(v):
    return "'" + str(v).replace("'", "''") + "'"


def num(v):
    v = str(v).strip()
    return None if v in ("", "nan", "None") else v


# draws upsert (batched: one giant statement exceeds the OS arg limit
# once history grew past a few hundred draws with the v2 backfill)
df = pd.read_csv(f"{BASE}/data/history.csv")
vals = [f"({esc(pd.Timestamp(r['date']).date().isoformat())}, {esc(r['draw'])}, "
        f"{int(r['d1'])}, {int(r['d2'])}, {int(r['d3'])})"
        for _, r in df.iterrows()]
BATCH = 200
for i in range(0, len(vals), BATCH):
    q("INSERT INTO public.draws (date, draw_type, d1, d2, d3) VALUES "
      + ", ".join(vals[i:i + BATCH]) +
      " ON CONFLICT (date, draw_type) DO UPDATE SET "
      "d1=EXCLUDED.d1, d2=EXCLUDED.d2, d3=EXCLUDED.d3;")
print(f"draws upserted in {(len(vals) + BATCH - 1) // BATCH} batches")

# guesses: insert new, update newly-scored
g = pd.read_csv(f"{BASE}/logs/guesses.csv", dtype=str)
# actuals lookup from history (local CSV has no a1/a2/a3 columns)
hist = {(pd.Timestamp(r["date"]).date().isoformat(), r["draw"]):
        (int(r["d1"]), int(r["d2"]), int(r["d3"]))
        for _, r in df.iterrows()}
for _, r in g.iterrows():
    date, draw = esc(r["date"]), esc(r["draw"])
    exists = json.dumps(q(
        f"SELECT id FROM public.guesses WHERE date={date} AND draw_type={draw} LIMIT 1;"))
    av = hist.get((r["date"], r["draw"]), (None, None, None))
    a1s, a2s, a3s = ("NULL" if v is None else str(v) for v in av)
    bh = num(r.get("box_hit"))
    dr = num(r.get("digits_right"))
    bh_sql = ("NULL" if bh is None else ("TRUE" if bh == "1" else "FALSE"))
    dr_sql = "NULL" if dr is None else dr
    if "id" not in exists:
        q("INSERT INTO public.guesses (date, draw_type, g1, g2, g3, a1, a2, a3, box_hit, digits_right) "
          f"VALUES ({date}, {draw}, {r['g1']}, {r['g2']}, {r['g3']}, "
          f"{a1s}, {a2s}, {a3s}, {bh_sql}, {dr_sql});")
        print("inserted", r["date"], r["draw"])
    elif bh is not None:
        q(f"UPDATE public.guesses SET a1={a1s}, a2={a2s}, a3={a3s}, "
          f"box_hit={bh_sql}, digits_right={dr_sql} "
          f"WHERE date={date} AND draw_type={draw} AND box_hit IS NULL;")
        print("scored", r["date"], r["draw"])

print("sync done")

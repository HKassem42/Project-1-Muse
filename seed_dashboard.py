"""Seed Lovable Lotto-Muse dashboard DB from local CSVs. Idempotent (upserts)."""
import json
import subprocess
import pandas as pd

PROJECT = "a1824c13-8733-445b-9f87-6d13929d7f94"


def q(sql):
    r = subprocess.run(
        ["/opt/hatch/bin/lovable", "call-tool", "--name", "query_database",
         "--arguments-json", json.dumps({"project_id": PROJECT, "sql": sql})],
        capture_output=True, text=True, timeout=60)
    d = json.loads(r.stdout)
    if not d.get("ok"):
        raise RuntimeError(r.stdout[:500])
    return d["result"].get("structuredContent")


def esc(v):
    return "'" + str(v).replace("'", "''") + "'"


# --- draws ---
df = pd.read_csv("/home/hatch/workspace/lotto-muse/data/history.csv")
vals = []
for _, r in df.iterrows():
    date = pd.Timestamp(r["date"]).date().isoformat()
    vals.append(f"({esc(date)}, {esc(r['draw'])}, {int(r['d1'])}, "
                f"{int(r['d2'])}, {int(r['d3'])})")
sql = ("INSERT INTO public.draws (date, draw_type, d1, d2, d3) VALUES "
       + ", ".join(vals) +
       " ON CONFLICT (date, draw_type) DO UPDATE SET "
       "d1=EXCLUDED.d1, d2=EXCLUDED.d2, d3=EXCLUDED.d3;")
print("draws upsert:", json.dumps(q(sql))[:200])

# --- guesses ---
g = pd.read_csv("/home/hatch/workspace/lotto-muse/logs/guesses.csv", dtype=str)
for _, r in g.iterrows():
    def col(c):
        v = r.get(c)
        return "NULL" if v is None or str(v).strip() in ("", "nan") else (
            "TRUE" if str(v) == "1" and c == "box_hit" else
            "FALSE" if str(v) == "0" and c == "box_hit" else v)
    sql = ("INSERT INTO public.guesses "
           "(date, draw_type, g1, g2, g3, a1, a2, a3, box_hit, digits_right) "
           f"VALUES ({esc(r['date'])}, {esc(r['draw'])}, {r['g1']}, {r['g2']}, {r['g3']}, "
           f"{col('a1')}, {col('a2')}, {col('a3')}, {col('box_hit')}, {col('digits_right')}) "
           "ON CONFLICT DO NOTHING;")
    # guesses has no unique constraint besides id; avoid dupes manually
    chk = q(f"SELECT id FROM public.guesses WHERE date={esc(r['date'])} "
            f"AND draw_type={esc(r['draw'])} LIMIT 1;")
    if "id" not in json.dumps(chk):
        q(sql)
        print("inserted guess", r["date"], r["draw"])
    else:
        print("guess exists", r["date"], r["draw"])

print("counts:", json.dumps(q("SELECT (SELECT COUNT(*) FROM public.draws) AS draws, "
                              "(SELECT COUNT(*) FROM public.guesses) AS guesses;"))[:200])

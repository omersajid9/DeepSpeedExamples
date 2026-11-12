#!/usr/bin/env python3
"""Phase-split kernel time from an nsys sqlite export (same rule as parse_trace_phases.py:
boundary = end of last kernel instance with duration >= threshold microseconds)."""
import sqlite3
import sys

db_file, threshold = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
con = sqlite3.connect(db_file)
cur = con.cursor()

# nsys sqlite: CUPTI_ACTIVITY_KIND_KERNEL (or CONCURRENT_KERNEL) with start/end in ns
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%KERNEL%'")]
print("kernel tables:", tables)
tab = "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL" if "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL" in tables else tables[0]

rows = cur.execute(f"""
  SELECT k.start, k.end, s.value FROM {tab} k
  JOIN StringIds s ON k.shortName = s.id""").fetchall()
print(f"{len(rows)} kernel instances")

tot = sum(e - s for s, e, _ in rows)
big = [(s, e, n) for s, e, n in rows if e - s >= threshold * 1000]
boundary = max(e for s, e, _ in big) if big else 0
print(f"total kernel time {tot/1e6:.1f} ms; prefill boundary at {boundary} ns")

import collections
agg = collections.defaultdict(lambda: [0, 0])
for s, e, n in rows:
    ph = "prefill" if s < boundary else "decode"
    a = agg[(ph, n)]
    a[0] += e - s
    a[1] += 1

for ph in ("prefill", "decode"):
    ptot = sum(v[0] for (p, _), v in agg.items() if p == ph)
    print(f"\n== {ph}: {ptot/1e6:.1f} ms")
    top = sorted(((n, v) for (p, n), v in agg.items() if p == ph), key=lambda kv: -kv[1][0])[:12]
    for n, (us, c) in top:
        print(f"   {us/1e6:8.1f} ms  x{c:<6d} {n[:80]}")

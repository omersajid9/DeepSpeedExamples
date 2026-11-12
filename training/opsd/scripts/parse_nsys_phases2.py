#!/usr/bin/env python3
"""Measured-iteration phase split from nsys sqlite, anchored on the last
36 prefill + 540 decode flash instances (36 layers; 6k: 15 decode steps)."""
import sqlite3
import sys
import collections

db_file = sys.argv[1]
flash_pat = sys.argv[2] if len(sys.argv) > 2 else "flash"
last_n = int(sys.argv[3]) if len(sys.argv) > 3 else 576
con = sqlite3.connect(db_file)
cur = con.cursor()

rows = cur.execute(f"""
  SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
  WHERE s.value LIKE '%{flash_pat}%'
  ORDER BY k.start""").fetchall()

n = len(rows)
print(f"{n} flash instances (pat={flash_pat!r}); using last {last_n}")
last = rows[-last_n:]
prefill_flash = last[:36]
decode_flash = last[36:]
iter_start = prefill_flash[0][0]
iter_end = max(r[1] for r in rows[-10:])
boundary = decode_flash[0][0]  # start of first decode flash
print(f"iteration window: {(iter_end - iter_start) / 1e6:.1f} ms; "
      f"prefill->decode boundary = start of first decode flash")

allk = cur.execute(f"""
  SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
  WHERE k.start >= {iter_start - 200_000} AND k.end <= {iter_end + 200_000}""").fetchall()
tot = sum(e - s for s, e, *_ in allk)
print(f"kernels in window: {len(allk)}, total {tot / 1e6:.1f} ms")

agg = collections.defaultdict(lambda: [0, 0])
grids = collections.defaultdict(collections.Counter)
for s, e, nm, gx, gy, gz, bx in allk:
    ph = "prefill" if s < boundary else "decode"
    a = agg[(ph, nm)]
    a[0] += e - s
    a[1] += 1
    if "flash" in nm:
        grids[(ph, nm)][f"({gx},{gy},{gz})x({bx},)"] += 1

for ph in ("prefill", "decode"):
    ptot = sum(v[0] for (p, _), v in agg.items() if p == ph)
    print(f"\n== {ph}: {ptot / 1e6:.1f} ms")
    top = sorted(((nm, v) for (p, nm), v in agg.items() if p == ph), key=lambda kv: -kv[1][0])[:12]
    for nm, (us, c) in top:
        print(f"   {us / 1e6:8.1f} ms  x{c:<6d} {nm[:70]}")
    for (p, nm), cnt in grids.items():
        if p == ph:
            print(f"   grid {nm[:40]}: {dict(cnt)}")

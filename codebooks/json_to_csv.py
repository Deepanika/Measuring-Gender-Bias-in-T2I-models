"""Flatten a role codebook (JSON array of scenario objects) into a review CSV, one row per role.
Usage: python json_to_csv.py codebook_v1.json            -> writes codebook_v1_review.csv
Works for any prompt version: every field present in the JSON becomes a column."""
import csv, json, sys
from pathlib import Path

src = Path(sys.argv[1])
data = json.loads(src.read_text(encoding="utf-8"))
rows = []
for s in data:
    scen = {k: v for k, v in s.items() if k != "roles"}
    for k in ("scene_questions", "derived_measures"):
        if k in scen:
            scen[k] = json.dumps(scen[k], ensure_ascii=False)
    for r in s.get("roles", []):
        rows.append({**scen, **{f"role_{k}" if k in scen else k: v for k, v in r.items()}})
cols = list(dict.fromkeys(k for row in rows for k in row))
out = src.with_name(src.stem + "_review.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
print(f"{len(data)} scenarios, {len(rows)} roles -> {out}")
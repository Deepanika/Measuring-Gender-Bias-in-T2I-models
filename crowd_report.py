
"""
crowd_report.py - how many images of each scenario are crowded, per T2I model and phrasing.

  python crowd_report.py vlm_results/results_gpt-6.1-sol_low_vlm_v1_main_images.csv ^
                         vlm_results/results_gemini-3.8-flash_low_vlm_v1_main_images.csv

Non-crowded = both VLMs coded the image OK and listed everyone individually (no crowd group), the same
"No crowd" definition as the agreement table. Each cell shows non-crowded images out of all images.
Flags: ALL CROWDED (no non-crowded image in the scenario), HAS EMPTY CELL (some model x phrasing has none),
MOSTLY CROWDED (under half of the scenario's images non-crowded).
Also saves the table as crowd_by_scenario.csv next to the first file.
"""

import json
import sys
from pathlib import Path

import pandas as pd

CODEBOOK = Path("codebook_final.json")


def load(path):
    d = pd.read_csv(path)
    d["exact_count"] = d["exact_count"].astype(str).str.lower().eq("true")
    return d.set_index("image_key")


def main(path_a, path_b):
    a, b = load(path_a), load(path_b)
    keys = a.index.intersection(b.index)
    d = a.loc[keys, ["t2i_model", "scenario_id", "phrasing"]].copy()
    d["code"] = [k.split("_")[1] for k in keys]
    d["non_crowded"] = (a.loc[keys, "run_status"].eq("ok") & b.loc[keys, "run_status"].eq("ok")
                        & a.loc[keys, "exact_count"] & b.loc[keys, "exact_count"]).values
    activity = {s["scenario_id"]: s["main_activity"] for s in json.loads(CODEBOOK.read_text(encoding="utf-8"))}

    rows = []
    for (sid, code), g in d.groupby(["scenario_id", "code"]):
        row = dict(scenario=sid, code=code)
        cells = []
        for m in sorted(d.t2i_model.unique()):
            for ph in ("plain", "use-case"):
                c = g[(g.t2i_model == m) & (g.phrasing == ph)]
                row[f"{m} {ph}"] = f"{int(c.non_crowded.sum())}/{len(c)}"
                cells.append(int(c.non_crowded.sum()))
        n_ok, n = int(g.non_crowded.sum()), len(g)
        row["total"] = f"{n_ok}/{n}"
        row["flag"] = ("ALL CROWDED" if n_ok == 0 else "HAS EMPTY CELL" if 0 in cells
                       else "MOSTLY CROWDED" if n_ok < n / 2 else "")
        row["description"] = activity.get(sid, "")
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("scenario", key=lambda s: s.str[1].astype(int) * 10 + s.str[3].astype(int))

    pd.set_option("display.width", 250)
    print(f"Non-crowded images: {int(d.non_crowded.sum())} of {len(d)}\n")
    print(t.drop(columns="description").to_string(index=False))
    flagged = t[t.flag != ""]
    print(f"\nFlagged scenarios: {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"  {r.scenario} ({r.code}) {r.flag}, {r.total} non-crowded: {r.description}")
    out = Path(path_a).parent / "crowd_by_scenario.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python crowd_report.py <vlm_A_images.csv> <vlm_B_images.csv>")
    main(sys.argv[1], sys.argv[2])

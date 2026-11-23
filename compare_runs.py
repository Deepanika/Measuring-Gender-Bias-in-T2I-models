"""
compare_runs.py - one robustness table across tus_compare.py runs: does each 3-digit finding survive a
different threshold, no filter at all, another coder, or the sensitivity roles?

  python compare_runs.py main n11 n15 strict_n7 all_images gpt_coder sensitivity

Each argument is a --tag you ran (folder vlm_results/tus_compare/<tag>/). The first is the reference (main).
Writes vlm_results/tus_compare/robustness/
  robustness_table.csv / .tex   per scenario x model: observed share and verdict in every run, and whether the
                                main finding holds in every run where the cell is measured
  filter_effect.csv             main (filtered) vs the run without filter: images kept and observed share per cell,
                                to show whether dropping crowded / large images changed the gender mix
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHORT = {"amplifies": "amp", "reduces": "red", "reverses": "rev", "consistent with TUS": "=",
         "departs from parity toward women": "W+", "departs from parity toward men": "M+"}


def verdict(r):
    if r.status != "measured":
        return "n/m"                                   
    if str(r.direction).startswith("negligible"):
        return "neg"
    return SHORT.get(r.direction, r.direction) + ("*" if r.significant_main else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--dir", default="vlm_results/tus_compare", type=Path)
    ap.add_argument("--unfiltered", default="all_images", help="tag of the run without the crowd / size filter")
    args = ap.parse_args()
    out = args.dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    runs = {}
    for t in args.tags:
        f = args.dir / t / "results.csv"
        if not f.exists():
            raise SystemExit(f"missing {f}: run tus_compare.py --tag {t} first")
        r = pd.read_csv(f)
        runs[t] = r[(r.level == "3-digit") & (r.comparison == "main")].set_index(["scenario", "model"])
    ref = args.tags[0]
    keys = runs[ref].index
    rows = []
    for k in keys:
        row = {"scenario": k[0], "model": k[1]}
        holds = []
        main_v = verdict(runs[ref].loc[k])
        for t, r in runs.items():
            if k not in r.index:
                row[t] = "n/m"
                continue
            x = r.loc[k]
            v = verdict(x)
            obs = "" if pd.isna(x.observed) else f"{x.observed:.0f}"
            row[t] = f"{obs} {v}".strip()
            if t != ref and main_v.endswith("*") and v != "n/m":
                holds.append(v == main_v)
        row["main finding"] = main_v
        row["holds in all measured runs"] = ("yes" if holds and all(holds) else "no" if holds else
                                             "" if not main_v.endswith("*") else "only main measured")
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "robustness_table.csv", index=False)
    print("Observed female share (%) and verdict per run. * = finding after correction; amp/red/rev = amplifies / "
          "reduces / reverses; W+ = more women where TUS is about even; = consistent; neg = under 5 pp; n/m = not measured")
    print(t.to_string(index=False))
    f = t[t["main finding"].str.endswith("*")]
    print(f"\nMain findings: {len(f)}; hold in every run where measured: {int((f['holds in all measured runs'] == 'yes').sum())}")
    nh = f[f["holds in all measured runs"] == "no"]
    if len(nh):
        print("Not robust in every run: " + ", ".join(f"{r.scenario} {r.model}" for r in nh.itertuples()))

    esc = lambda s: str(s).replace("*", "$^{*}$").replace("%", r"\%").replace("_", r"\_")
    cols = ["scenario", "model"] + args.tags + ["holds in all measured runs"]
    body = "\n".join(" & ".join(esc(r[c]) for c in cols) + r" \\" for _, r in t.iterrows())
    tex = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n"
           f"\\begin{{tabular}}{{@{{}}ll{'c' * (len(args.tags) + 1)}@{{}}}}\n\\toprule\n"
           + " & ".join(esc(c) for c in cols) + " \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}}\n\\caption{Robustness of the 3-digit comparisons across runs. "
           "[Write your caption here.]}\n\\label{tab:robustness}\n\\end{table}\n")
    (out / "robustness_table.tex").write_text(tex, encoding="utf-8")

    if args.unfiltered in runs:
        a, b = runs[ref], runs[args.unfiltered]
        fe = pd.DataFrame({
            "kept_filtered": a.images_kept, "images_total": b.images_kept,
            "share_filtered": a.observed, "share_unfiltered": b.reindex(a.index).observed})
        fe["kept_pct"] = (100 * fe.kept_filtered / fe.images_total).round(0)
        fe["share_change_pp"] = (fe.share_filtered - fe.share_unfiltered).round(1)
        fe = fe.reset_index()
        fe.to_csv(out / "filter_effect.csv", index=False)
        both = fe.dropna(subset=["share_filtered", "share_unfiltered"])
        print(f"\nFILTER EFFECT (main vs {args.unfiltered}): cells measured in both: {len(both)}; "
              f"median |change in female share| {both.share_change_pp.abs().median():.1f} pp, "
              f"max {both.share_change_pp.abs().max():.1f} pp")
        print("Cells that lost at least 10% of their images:")
        print(both[both.kept_pct <= 90][["scenario", "model", "kept_filtered", "images_total", "share_filtered",
                                          "share_unfiltered", "share_change_pp"]].to_string(index=False))
        by = fe.groupby("model")[["kept_filtered", "images_total"]].sum()
        print("Images kept by the filter per T2I model: " +
              ", ".join(f"{m} {int(r.kept_filtered)}/{int(r.images_total)}" for m, r in by.iterrows()))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
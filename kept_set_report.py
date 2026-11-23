"""
kept_set_report.py - what is left to analyse at a people-per-image limit n, and how well the two VLMs
agree on exactly those images.

  python kept_set_report.py --n 13 7

Kept image = both VLMs coded it OK, neither used a crowd group (non-crowded), and the PRIMARY VLM listed
at most n people (the same rule as section 6 of choose_threshold.py and as the main analysis).

For each n (outputs in vlm_results/kept_set/):
  1 totals: kept of all images, by T2I model, by phrasing
  2 every scenario x model x phrasing cell (20 images each): images kept; EMPTY and THIN (< half) cells
    and what each scenario can support: both models / one model only / descriptive only   cells_n<N>.csv
  3 VLM vs VLM agreement on the kept images, overall and by T2I model and phrasing     agreement_n<N>.csv/.tex
    (same measures as the agreement table: % identical / % within one person; majority gender: % (kappa))
  4 per scenario x model: VLM vs VLM agreement on the kept images (counts and roles) with a 95% interval;
    cells below --caution % are flagged "interpret with caution" (flagged, not removed)  cell_agreement_n<N>.csv
  5 per scenario x model: how alike the images are in size (all images, primary VLM): share crowded,
    people per image (min, quartiles, max), share above n                              cell_spread_n<N>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from validate_humans import flat, kappa_str, tex
from vlm_agreement_basic import majority_gender, pct_identical, pct_within_one


def fmt(a, b):
    return f"{pct_identical(a, b):.1f} / {pct_within_one(a, b):.1f}" if len(a) else "--"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def role_same(P, S, keys):
    """per image: True if every role category has the same number of women and of men for both coders"""
    ra = P["roles"][P["roles"].image_key.isin(keys)].groupby(["image_key", "category"])[["n_female", "n_male"]].sum()
    rb = S["roles"][S["roles"].image_key.isin(keys)].groupby(["image_key", "category"])[["n_female", "n_male"]].sum()
    c = ra.join(rb, how="outer", lsuffix="_A", rsuffix="_B").fillna(0)
    same = ((c.n_female_A == c.n_female_B) & (c.n_male_A == c.n_male_B)).groupby(level=0).all()
    return same.reindex(keys, fill_value=True)


def agreement(P, S, keys):
    a, b = P["img"].loc[keys], S["img"].loc[keys]
    out = {"Images": str(len(keys))}
    for lab, v in [("People per image", "n_people_listed"), ("Women per image", "n_female"), ("Men per image", "n_male")]:
        out[lab] = fmt(a[v].values, b[v].values)
    both = (a.n_female.values == b.n_female.values) & (a.n_male.values == b.n_male.values)
    out["Women and men both identical (%)"] = f"{100 * both.mean():.1f}" if len(keys) else "--"
    keep = ["image_key", "category", "n_female", "n_male"]
    ra = P["roles"][P["roles"].image_key.isin(keys)].groupby(["image_key", "category"])[["n_female", "n_male"]].sum()
    rb = S["roles"][S["roles"].image_key.isin(keys)].groupby(["image_key", "category"])[["n_female", "n_male"]].sum()
    c = ra.join(rb, how="outer", lsuffix="_A", rsuffix="_B").fillna(0)
    for lab, g in (("Women per role category", "n_female"), ("Men per role category", "n_male")):
        s = c[(c[g + "_A"] > 0) | (c[g + "_B"] > 0)]
        out[lab] = fmt(s[g + "_A"].values, s[g + "_B"].values)
    for f, name in (("foreground", "Foreground"), ("focal", "Focal")):
        m = pd.DataFrame({f"{f}_n_female_A": a[f"{f}_n_female"].values, f"{f}_n_male_A": a[f"{f}_n_male"].values,
                          f"{f}_n_female_B": b[f"{f}_n_female"].values, f"{f}_n_male_B": b[f"{f}_n_male"].values})
        out[f"{name}: majority gender"] = kappa_str(majority_gender(m, f, "A"), majority_gender(m, f, "B"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[13, 7])
    ap.add_argument("--primary", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--secondary", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--out-dir", default="vlm_results/kept_set", type=Path)
    ap.add_argument("--caution", type=float, default=80.0,
                    help="flag scenario x model cells whose VLM-VLM count agreement on kept images is below this %%")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    P, S = flat(args.primary), flat(args.secondary)
    pn, sn = (str(x["img"]["vlm_model_requested"].iloc[0]) for x in (P, S))
    both = P["img"].index.intersection(S["img"].index)
    d = P["img"].loc[both, ["scenario_id", "t2i_model", "phrasing", "n_people_listed"]].copy()
    d["code"] = [k.split("_")[1] for k in both]
    d["noncrowd"] = (P["img"].loc[both, "exact_count"].values & S["img"].loc[both, "exact_count"].values)
    order = sorted(d.scenario_id.unique(), key=lambda s: int(s[1]) * 10 + int(s[3]))
    models, phrasings = sorted(d.t2i_model.unique()), ["plain", "use-case"]
    print(f"{len(both)} images coded by both {pn} (primary, decides size) and {sn}; "
          f"{int(d.noncrowd.sum())} non-crowded")

    for n in args.n:
        d["kept"] = d.noncrowd & (d.n_people_listed <= n)
        keys = list(d.index[d.kept])
        print(f"\n================ n = {n} ================")
     
        print(f"1. KEPT: {len(keys)} of {len(d)} images ({100 * len(keys) / len(d):.1f}%)")
        for col in ("t2i_model", "phrasing"):
            t = d.groupby(col).kept.agg(["sum", "size"])
            print("   by " + col + ": " + " | ".join(f"{i} {int(r['sum'])}/{int(r['size'])}" for i, r in t.iterrows()))
        t = d.groupby(["t2i_model", "phrasing"]).kept.agg(["sum", "size"])
        print("   by model x phrasing: " + " | ".join(f"{m} {p} {int(r['sum'])}/{int(r['size'])}" for (m, p), r in t.iterrows()))

      
        rows = []
        for s in order:
            g = d[d.scenario_id == s]
            row = {"scenario": s, "code": g.code.iloc[0]}
            for m in models:
                for p in phrasings:
                    c = g[(g.t2i_model == m) & (g.phrasing == p)]
                    row[f"{m} {p}"] = f"{int(c.kept.sum())}/{len(c)}"
                gm = g[g.t2i_model == m]
                row[f"{m} total"] = int(gm.kept.sum())
                row[f"_{m}_ok"] = gm.kept.mean() >= 0.5
            ok = [m for m in models if row[f"_{m}_ok"]]
            row["supports"] = ("both models" if len(ok) == len(models) else
                               f"{ok[0]} only" if ok else "descriptive only")
            rows.append(row)
        cells = pd.DataFrame(rows)
        cells.drop(columns=[c for c in cells if c.startswith("_")]).to_csv(args.out_dir / f"cells_n{n}.csv", index=False)
        print(f"\n2. IMAGES KEPT PER CELL (out of 20 per model x phrasing); a model needs at least half of its 40 "
              f"images kept in a scenario to be measured there")
        print(cells.drop(columns=[c for c in cells if c.startswith("_")]).to_string(index=False))
        empty = [f"{r['scenario']} {m} {p}" for _, r in cells.iterrows() for m in models for p in phrasings
                 if r[f"{m} {p}"].startswith("0/")]
        print(f"   EMPTY cells (0 images kept): {len(empty)} -> {', '.join(empty) or 'none'}")
        for lab in ("both models", "descriptive only"):
            sel = cells[cells.supports == lab].scenario.tolist()
            print(f"   {lab}: {len(sel)} scenarios {', '.join(sel)}")
        one = cells[cells.supports.str.endswith(" only") & (cells.supports != "descriptive only")]
        print(f"   one model only: {len(one)} scenarios " + ", ".join(f"{r.scenario} ({r.supports})" for r in one.itertuples()))

     
        res = {"All kept": agreement(P, S, keys)}
        for m in models:
            res[f"{m} images"] = agreement(P, S, [k for k in keys if d.at[k, "t2i_model"] == m])
        for p in phrasings:
            res[p] = agreement(P, S, [k for k in keys if d.at[k, "phrasing"] == p])
        ag = pd.DataFrame(res)
        ag.to_csv(args.out_dir / f"agreement_n{n}.csv")
        print(f"\n3. {pn} vs {sn} ON THE KEPT IMAGES ('% identical / % within one'; majority gender '% (kappa)')")
        print(ag.to_string())
        (args.out_dir / f"agreement_n{n}.tex").write_text(tex(
            [[i] + list(r) for i, r in ag.iterrows()], list(ag.columns),
            f"Agreement between the two VLM coders on the images analysed at n = {n}.", f"tab:kept-agreement-n{n}"),
            encoding="utf-8")

      
        pa, sa = P["img"], S["img"]
        rsame = role_same(P, S, keys)
        rows = []
        for s_ in order:
            for m in models:
                ks = [k for k in keys if d.at[k, "scenario_id"] == s_ and d.at[k, "t2i_model"] == m]
                if not ks:
                    rows.append(dict(scenario=s_, t2i_model=m, kept=0, counts_same="--", counts_95ci="--",
                                     roles_same="--", flag="no images kept"))
                    continue
                same = ((pa.loc[ks, "n_female"].values == sa.loc[ks, "n_female"].values) &
                        (pa.loc[ks, "n_male"].values == sa.loc[ks, "n_male"].values))
                k_, n_ = int(same.sum()), len(ks)
                lo, hi = wilson(k_, n_)
                pc = 100 * k_ / n_
                flag = ("too few images to judge" if n_ < 5 else
                        "interpret with caution" if pc < args.caution else "")
                rows.append(dict(scenario=s_, t2i_model=m, kept=n_, counts_same=round(pc, 1),
                                 counts_95ci=f"{lo:.0f}-{hi:.0f}", roles_same=round(100 * rsame.loc[ks].mean(), 1),
                                 flag=flag))
        ca = pd.DataFrame(rows)
        ca.to_csv(args.out_dir / f"cell_agreement_n{n}.csv", index=False)
        print(f"\n4. PER SCENARIO x MODEL: % of kept images where {pn} and {sn} give identical women and men counts "
              f"(95% interval) and identical counts in every role category; flagged below {args.caution:g}%")
        print(ca.to_string(index=False))
        fl = ca[ca.flag == "interpret with caution"]
        print(f"   Flagged 'interpret with caution': {len(fl)} cells " +
              ", ".join(f"{r.scenario} {r.t2i_model} ({r.counts_same}%)" for r in fl.itertuples()))

        
        rows = []
        for s_ in order:
            for m in models:
                g = d[(d.scenario_id == s_) & (d.t2i_model == m)]
                ppl = g.n_people_listed
                nc = g[g.noncrowd].n_people_listed
                rows.append(dict(scenario=s_, t2i_model=m, images=len(g), crowded_pct=round(100 * (1 - g.noncrowd.mean()), 0),
                                 people_min=int(ppl.min()), people_q1=ppl.quantile(.25), people_median=ppl.median(),
                                 people_q3=ppl.quantile(.75), people_max=int(ppl.max()),
                                 noncrowded_above_n_pct=round(100 * (nc > n).mean(), 0) if len(nc) else np.nan))
        sp = pd.DataFrame(rows)
        sp.to_csv(args.out_dir / f"cell_spread_n{n}.csv", index=False)
        print(f"\n5. PER SCENARIO x MODEL: how alike the images are in size (all images; people listed by {pn}; "
              "crowded images list at most 20 people, so their counts are lower bounds)")
        print(sp.to_string(index=False))
        wide = sp[(sp.people_q3 - sp.people_q1 >= 4) | (sp.crowded_pct.between(10, 90))]
        print(f"   Mixed-size cells (quartiles 4+ people apart, or 10-90% crowded): {len(wide)} of {len(sp)}")
    print(f"\nOutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
"""
validate_humans.py - compare the four coders of the 108-image validation sample with each other.

  python validate_humans.py

Four coders: Main (C01, every image), Vol (the one volunteer per image, C02-C09 together), and the two
VLMs A and B. Six pairs: Main-Vol (human-human), A-Main, A-Vol, B-Main, B-Vol (human-VLM), A-B (VLM-VLM).
Everything is computed on the 100 NON-CROWDED images, except section 7 (the 8 crowded fallbacks).
Sensitivity runs (outputs in a subfolder, the full results stay as they are):
  --large-scenes        leave out large-group scenarios (rule fixed from the VLM crowd data, printed)
  --exclude-scenarios   leave out scenarios you name;  --max-people N  leave out images with more than N people

Reads (change with options):
  vlm_results/results_C01_human_vlm_v1_main.jsonl, vlm_results/results_C0[2-9]_human_vlm_v1_main.jsonl
  vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl, vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl
  private/selection.csv (image_type marks the crowded fallbacks), codebook_final.json

Writes to vlm_results/human_validation/  (.tex tables have a [Write your caption here.] placeholder)
  1 validation_table.tex/.csv   per image: people, women, men, women/men per role and per role category,
                                foreground and focal (type, size, majority gender), scene question
  2 category_agreement.csv/.tex per role category (subtypes split): % of images with the same number of
                                women / men in that category, for every pair
  3 category_shares.csv/.tex    per coder: women, men and female share in each category
    category_share_diff.csv     female share within each category, coder minus reference, 95% bootstrap
                                interval, all images and by T2I model (does any coder shift women between roles?)
  4 female_share.csv/.tex       overall female share, coder minus reference, 95% interval, by T2I model
  5 person_matches.csv          humans only: people matched by marker position; gender, role and category of each
    person_confusion.csv        role category given by Main (rows) vs Vol (columns) to the same person
  6 per_volunteer.csv, per_scenario.csv   where disagreement concentrates
  7 crowded_prominence.csv      crowded fallbacks: majority gender of foreground and focal, every coder
  9 by_image_model.csv          agreement on each T2I model's images separately
 10 category_by_scenario.csv    women/people per role category, scenario and coder
 11 unclear_use.csv             how often each coder used unclear gender, unclear role, none of the above
  8 adjudication_queue.csv      non-crowded images where the humans differ in the number of women or men;
    gold_counts.csv/.tex        once every row is decided: each coder against the adjudicated counts
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import flatten_results
from vlm_agreement_basic import cohen_kappa, majority_gender, pct_identical, pct_within_one

BOOT, SEED = 2000, 2026
warnings.filterwarnings("ignore", category=RuntimeWarning)  
CATEGORY_ORDER = ["PRIMARY", "DIRECTING", "ASSISTING", "RECEIVING", "OTHER_TASK:paid_work",
                  "OTHER_TASK:domestic_chore", "OTHER_TASK:care", "OBSERVING:watching", "OBSERVING:own_activity",
                  "UNCLEAR_ROLE", "OTHER_UNLISTED"]



def flat(jsonl):
    """Flattened tables of a results file; reruns flatten_results.py when they are missing or older."""
    jsonl = Path(jsonl)
    stem = jsonl.with_suffix("")
    files = {n: Path(f"{stem}_{n}.csv") for n in ("images", "role_counts", "people")}
    if any(not f.exists() or f.stat().st_mtime < jsonl.stat().st_mtime for f in files.values()):
        flatten_results.main(jsonl)
    i, r, p = (pd.read_csv(files[n]) for n in ("images", "role_counts", "people"))
    i["exact_count"] = i["exact_count"].astype(str).str.lower().eq("true")
    for d in (r, p):
        d["group"] = [c if s in ("none", "") or pd.isna(s) else f"{c}:{s}" for c, s in zip(d.category, d.subtype)]
    return dict(img=i[i.run_status == "ok"].set_index("image_key", drop=False), roles=r, people=p)


def records(jsonl):
    return [json.loads(l) for l in Path(jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt(a, b):
    return f"{pct_identical(a, b):.1f} / {pct_within_one(a, b):.1f}" if len(a) else "--"


def kappa_str(a, b):
    if not len(a):
        return "--"
    k = cohen_kappa(a, b)
    return f"{pct_identical(a, b):.1f} ({'--' if np.isnan(k) else f'{k:.2f}'})"


def tex(rows, cols, caption, label, first="", size="scriptsize"):
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    head = " & ".join([esc(first)] + [rf"\textbf{{{esc(c)}}}" for c in cols]) + r" \\"
    body = "\n".join(" & ".join(esc(x) for x in r) + r" \\" for r in rows)
    rb0, rb1 = (r"\resizebox{\textwidth}{!}{%", "}") if len(cols) > 6 else ("", "")   
    return (rf"""\begin{{table}}[t]
\centering
\{size}
{rb0}
\begin{{tabular}}{{@{{}}l{"c" * len(cols)}@{{}}}}
\toprule
{head}
\midrule
{body}
\bottomrule
\end{{tabular}}{rb1}
\caption{{{caption} [Write your caption here.]}}
\label{{{label}}}
\end{{table}}
""")


def cell_counts(roles, keys, by):
    """women / men per image x role (by='role_id') or per image x category group (by='group')."""
    r = roles[roles.image_key.isin(keys)]
    return r.groupby(["image_key", by], as_index=False)[["n_female", "n_male"]].sum()


def image_level(X, Y, keys):
    """Per-image agreement between coders X and Y (as in the agreement table, plus extra rows)."""
    ix, iy = X["img"].loc[keys], Y["img"].loc[keys]
    out = {"Images": str(len(keys))}
    for lab, v in [("People per image", "n_people_listed"), ("Women per image", "n_female"),
                   ("Men per image", "n_male"), ("Unclear-gender people per image", "n_unclear")]:
        out[lab] = fmt(ix[v], iy[v])
    for lab, by in [("role", "role_id"), ("role category", "group_cat")]:
        rx, ry = X["roles"].copy(), Y["roles"].copy()
        rx["group_cat"], ry["group_cat"] = rx["category"], ry["category"]
        c = pd.merge(cell_counts(rx, keys, by), cell_counts(ry, keys, by), on=["image_key", by],
                     how="outer", suffixes=("_x", "_y")).fillna(0)
        for g, name in (("n_female", "Women"), ("n_male", "Men")):
            s = c[(c[g + "_x"] > 0) | (c[g + "_y"] > 0)]
            out[f"{name} per {lab}"] = fmt(s[g + "_x"], s[g + "_y"])
    for f, name in (("foreground", "Foreground"), ("focal", "Focal")):
        out[f"{name}: type"] = kappa_str(ix[f"{f}_type"].astype(str).values, iy[f"{f}_type"].astype(str).values)
        out[f"{name}: number of people"] = fmt(ix[f"{f}_n"], iy[f"{f}_n"])
        m = pd.DataFrame({f"{f}_n_female_A": ix[f"{f}_n_female"].values, f"{f}_n_male_A": ix[f"{f}_n_male"].values,
                          f"{f}_n_female_B": iy[f"{f}_n_female"].values, f"{f}_n_male_B": iy[f"{f}_n_male"].values})
        out[f"{name}: majority gender"] = kappa_str(majority_gender(m, f, "A"), majority_gender(m, f, "B"))
    sq = [c for c in ix.columns if c.startswith("scene_q") and not c.endswith("_question")]
    for q in sq:
        both = ix[q].notna() & iy[q].notna()
        out[f"Scene question {q[6:]}"] = kappa_str(ix.loc[both, q].astype(str).values, iy.loc[both, q].astype(str).values) \
            + f" n={int(both.sum())}"
    return out


def pooled(f, m):
    t = f.sum(axis=-1) + m.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t > 0, f.sum(axis=-1) / t, np.nan)


def share_diff(fx, mx, fy, my, rng):
    """Pooled female share of X minus Y over the same images, with a 95% bootstrap interval."""
    n = len(fx)
    if n == 0 or (fx.sum() + mx.sum() + fy.sum() + my.sum()) == 0:
        return dict(images=n, women_x=0, men_x=0, women_y=0, men_y=0, share_x=np.nan, share_y=np.nan,
                    diff_pp=np.nan, ci_low=np.nan, ci_high=np.nan)
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = pooled(fx[idx], mx[idx]) - pooled(fy[idx], my[idx])
    sx, sy = pooled(fx, mx), pooled(fy, my)
    r = lambda v: round(100 * float(v), 1) if not np.isnan(v) else np.nan
    return dict(images=n, women_x=int(fx.sum()), men_x=int(mx.sum()), women_y=int(fy.sum()), men_y=int(my.sum()),
                share_x=r(sx), share_y=r(sy), diff_pp=r(sx - sy), ci_low=r(np.nanpercentile(boot, 2.5)),
                ci_high=r(np.nanpercentile(boot, 97.5)))


def group_arrays(C, keys, group):
    """women and men per image in one category group, aligned to keys."""
    r = C["roles"]
    r = r[r.image_key.isin(keys) & (r.group == group)].groupby("image_key")[["n_female", "n_male"]].sum()
    r = r.reindex(keys, fill_value=0)
    return r.n_female.to_numpy(float), r.n_male.to_numpy(float)


def match_people(a, b, threshold):
    """Greedy nearest-marker matching of two humans' people lists; returns pairs (i, j, distance)."""
    d = sorted((math.hypot(p["x"] - q["x"], p["y"] - q["y"]), i, j) for i, p in enumerate(a) for j, q in enumerate(b))
    used_a, used_b, pairs = set(), set(), []
    for dist, i, j in d:
        if dist > threshold:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, dist))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="vlm_results", type=Path)
    ap.add_argument("--main-coder", default="C01")
    ap.add_argument("--vlm-a", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--vlm-b", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--extra-vlm", action="append", type=Path, default=None,
                    help="more VLM results files to compare (repeatable); default: "
                         "vlm_results/results_gemini-4-argon_medium_vlm_v1_main.jsonl if it exists")
    ap.add_argument("--selection", default="private/selection.csv", type=Path)
    ap.add_argument("--codebook", default="codebook_final.json", type=Path)
    ap.add_argument("--large-scenes", action="store_true",
                    help="leave out large-group scenarios: those where either T2I model drew crowds in more than half "
                         "of its images (decided from the VLM crowd data, not from the human comparison)")
    ap.add_argument("--exclude-scenarios", default="", help="comma-separated scenario ids to leave out, e.g. D5S1,D7S2")
    ap.add_argument("--max-people", type=int, default=0,
                    help="leave out images where the main human coder listed more than this many people")
    ap.add_argument("--match-distance", type=float, default=0.20,
                    help="max marker distance (fraction of image width) to treat two humans' markers as one person")
    args = ap.parse_args()
    out = args.results_dir / "human_validation"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)


    main_file = args.results_dir / f"results_{args.main_coder}_human_vlm_v1_main.jsonl"
    vol_files = sorted(f for f in args.results_dir.glob("results_*_human_vlm_v1_main.jsonl") if f != main_file)
    vol_recs = [r for f in vol_files for r in records(f)]
    seen = pd.Series([r["image_key"] for r in vol_recs])
    if seen.duplicated().any():
        raise SystemExit(f"Images coded by two volunteers: {sorted(set(seen[seen.duplicated()]))}")
    vol_file = out / "results_volunteers_human_vlm_v1_main.jsonl"
    vol_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in vol_recs), encoding="utf-8")
    main_recs = {r["image_key"]: r for r in records(main_file)}
    vol_by = {r["image_key"]: r for r in vol_recs}

    extra = args.extra_vlm
    if extra is None:
        d = args.results_dir / "results_gemini-4-argon_medium_vlm_v1_main.jsonl"
        extra = [d] if d.exists() else []
    vlm_files = [("A", args.vlm_a), ("B", args.vlm_b)] + [(chr(ord("C") + i), f) for i, f in enumerate(extra)]
    C = {"Main": flat(main_file), "Vol": flat(vol_file)}
    name = {"Main": "Main", "Vol": "Vol"}
    for key, f in vlm_files:
        C[key] = flat(f)
        name[key] = str(C[key]["img"]["vlm_model_requested"].iloc[0])
        cbv = set(C[key]["img"]["codebook_version"].dropna()) | set(C[key]["img"]["vlm_prompt_version"].dropna())
        if not {"codebook_final@58a2c8633d", "vlm_v1"} >= cbv:
            print(f"WARNING: {f.name} uses {sorted(cbv)}, not codebook_final@58a2c8633d / vlm_v1")
    if len(set(name.values())) < len(name):                   
        for key, f in vlm_files:
            name[key] += f" ({C[key]['img']['reasoning_effort'].iloc[0]})"
    VLMS = [k for k, _ in vlm_files]
    ALL = ["Main", "Vol"] + VLMS
    PAIRS = [("Main", "Vol")] + [(v, h) for v in VLMS for h in ("Main", "Vol")] + \
            [(x, y) for i, x in enumerate(VLMS) for y in VLMS[i + 1:]]
    plabel = lambda x, y: f"{name[x]} vs {name[y]}"

    sel = pd.read_csv(args.selection)
    crowded = set(sel.loc[sel["image_type"] == "crowded_fallback", "image_key"]) if "image_type" in sel else set()
    coded = set.intersection(*(set(c["img"].index) for c in C.values()))
    keys = sorted((set(sel.image_key) - crowded) & coded)

    dropped, tags = [], []
    if args.large_scenes:
        both = C["A"]["img"].index.intersection(C["B"]["img"].index)
        d = C["A"]["img"].loc[both, ["scenario_id", "t2i_model"]].copy()
        d["crowded"] = ~(C["A"]["img"].loc[both, "exact_count"] & C["B"]["img"].loc[both, "exact_count"])
        frac = d.groupby(["scenario_id", "t2i_model"])["crowded"].mean().unstack()
        large = sorted(frac.index[(frac > 0.5).any(axis=1)])
        print("Large-group scenarios (a T2I model drew crowds in over half its images): " + ", ".join(
            f"{s} ({', '.join(f'{m} {frac.at[s, m]:.0%}' for m in frac.columns)})" for s in large))
        dropped += large
        tags.append("large-scenes")
    if args.exclude_scenarios:
        dropped += [x.strip() for x in args.exclude_scenarios.split(",") if x.strip()]
        tags.append("scen-" + "-".join(x.strip() for x in args.exclude_scenarios.split(",") if x.strip()))
    if dropped or args.max_people:
        before = len(keys)
        keys = [k for k in keys if C["Main"]["img"].at[k, "scenario_id"] not in set(dropped)]
        if args.max_people:
            keys = [k for k in keys if C["Main"]["img"].at[k, "n_people_listed"] <= args.max_people]
            tags.append(f"max{args.max_people}people")
        crowded = {k for k in crowded if C["Main"]["img"].at[k, "scenario_id"] not in set(dropped)}
        print(f"SUBSET: {len(keys)} of {before} non-crowded images kept "
              f"(left out: scenarios {', '.join(sorted(set(dropped))) or 'none'}"
              f"{f'; images with more than {args.max_people} people (main coder)' if args.max_people else ''})")
    res_dir = out / ("subset_" + "_".join(tags)) if tags else out
    res_dir.mkdir(parents=True, exist_ok=True)
    sizes = C["Main"]["img"].loc[keys, "n_people_listed"]
    print(f"People per image (main coder), {len(keys)} images: median {sizes.median():.0f}, "
          f"<=5: {(sizes <= 5).sum()}, 6-10: {((sizes > 5) & (sizes <= 10)).sum()}, >10: {(sizes > 10).sum()}")

    t2i = C["Main"]["img"]["t2i_model"]
    subsets = [("all", keys)] + [(m, [k for k in keys if t2i[k] == m]) for m in sorted(t2i[keys].unique())]
    scen_of = C["Main"]["img"]["scenario_id"]
    print(f"Validation sample: {len(sel)} images -> {len(keys)} non-crowded coded by all four, {len(crowded)} crowded")
    print("Coders: Main =", args.main_coder, "| Vol =", ", ".join(f.name.split("_")[1] for f in vol_files),
          "| " + " | ".join(f"{k} = {name[k]}" for k in VLMS))
    hc = [k for k in keys if not (C["Main"]["img"].at[k, "exact_count"] and C["Vol"]["img"].at[k, "exact_count"])]
    if hc:
        print(f"Note: a human used a crowd group in {len(hc)} non-crowded images; listed counts leave those people out.")

    # ---- 1. image-level table
    res = {plabel(x, y): image_level(C[x], C[y], keys) for x, y in PAIRS}
    t1 = pd.DataFrame(res)
    t1.to_csv(res_dir / "validation_table.csv")
    print(f"\n1. PER IMAGE ({len(keys)} non-crowded images): counts '% identical / % within one'; "
          "type, majority gender and scene question '% identical (kappa)'")
    print(t1.to_string())
    (res_dir / "validation_table.tex").write_text(tex([[i] + list(r) for i, r in t1.iterrows()], list(t1.columns),
                                                  f"Agreement per image on the non-crowded validation images (n = {len(keys)}).",
                                                  "tab:human-validation"), encoding="utf-8")


    groups = [g for g in CATEGORY_ORDER if any((c["roles"].group == g).any() for c in C.values())]
    rows = []
    for g in groups:
        for x, y in PAIRS:
            fx, mx = group_arrays(C[x], keys, g)
            fy, my = group_arrays(C[y], keys, g)
            row = dict(category=g, pair=plabel(x, y))
            for lab, a, b in (("women", fx, fy), ("men", mx, my)):
                cell = (a > 0) | (b > 0)
                row[f"{lab}_cells"] = int(cell.sum())
                row[f"{lab}_identical"] = round(pct_identical(a[cell], b[cell]), 1) if cell.any() else np.nan
                row[f"{lab}_within1"] = round(pct_within_one(a[cell], b[cell]), 1) if cell.any() else np.nan
            rows.append(row)
    ca = pd.DataFrame(rows)
    ca.to_csv(res_dir / "category_agreement.csv", index=False)
    for lab in ("women", "men"):
        piv = ca.pivot(index="category", columns="pair", values=f"{lab}_identical").reindex(groups)
        piv = piv[[plabel(x, y) for x, y in PAIRS]]
        n = ca.pivot(index="category", columns="pair", values=f"{lab}_cells").reindex(groups)[plabel("Main", "Vol")]
        piv.insert(0, "images (Main-Vol)", n)
        print(f"\n2. PER ROLE CATEGORY, {lab.upper()}: % of images with the same number of {lab} in the category "
              "(images where either coder placed someone there)")
        print(piv.to_string())
        if lab == "women":
            twomen = piv
    (res_dir / "category_agreement.tex").write_text(tex(
        [[g] + [("--" if pd.isna(v) else f"{v:.1f}") for v in twomen.loc[g].values[1:]] for g in groups],
        [plabel(x, y) for x, y in PAIRS], "Per role category: \\% of images with the same number of women.",
        "tab:category-agreement", first="Category"), encoding="utf-8")


    srows, drows = [], []
    for g in groups:
        for c in ALL:
            f, m = group_arrays(C[c], keys, g)
            srows.append(dict(category=g, coder=name[c], women=int(f.sum()), men=int(m.sum()),
                              female_share=round(100 * float(pooled(f, m)), 1) if f.sum() + m.sum() else np.nan))
        for x, y in PAIRS:
            for sub, ks in subsets:
                fx, mx = group_arrays(C[x], ks, g)
                fy, my = group_arrays(C[y], ks, g)
                drows.append(dict(category=g, pair=plabel(x, y), t2i_images=sub, **share_diff(fx, mx, fy, my, rng)))
    cs, cd = pd.DataFrame(srows), pd.DataFrame(drows)
    cs.to_csv(res_dir / "category_shares.csv", index=False)
    cd.to_csv(res_dir / "category_share_diff.csv", index=False)
    piv = cs.pivot(index="category", columns="coder", values="female_share").reindex(groups)[[name[c] for c in ALL]]
    cnt = cs.pivot(index="category", columns="coder", values="women").reindex(groups)[[name[c] for c in ALL]].astype(str) \
        + "/" + cs.assign(t=cs.women + cs.men).pivot(index="category", columns="coder", values="t").reindex(groups)[[name[c] for c in ALL]].astype(str)
    print("\n3. FEMALE SHARE WITHIN EACH CATEGORY, by coder (%), and women/people placed there")
    print(piv.join(cnt, rsuffix=" (women/people)").to_string())
    sig = cd[(cd.t2i_images == "all") & cd.ci_low.notna() & ((cd.ci_low > 0) | (cd.ci_high < 0))]
    print("   Pairs whose difference in a category's female share excludes 0 (all images):")
    print("   " + ("none" if sig.empty else "\n   ".join(f"{r.category}: {r.pair} {r.diff_pp:+.1f} pp [{r.ci_low:+.1f}, {r.ci_high:+.1f}]"
                                                      for r in sig.itertuples())))
    (res_dir / "category_shares.tex").write_text(tex(
        [[g] + [("--" if pd.isna(v) else f"{v:.1f}") for v in piv.loc[g].values] for g in groups],
        [name[c] for c in ALL], "Female share (\\%) within each role category, by coder.",
        "tab:category-shares", first="Category"), encoding="utf-8")

    
    frows, ftex = [], []
    for x, y in PAIRS:
        cells = [plabel(x, y)]
        for sub, ks in subsets:
            ix, iy = C[x]["img"].loc[ks], C[y]["img"].loc[ks]
            r = share_diff(ix.n_female.to_numpy(float), ix.n_male.to_numpy(float),
                           iy.n_female.to_numpy(float), iy.n_male.to_numpy(float), rng)
            frows.append(dict(pair=plabel(x, y), t2i_images=sub, **r))
            cells.append(f"{r['diff_pp']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]")
        ftex.append(cells)
    fs = pd.DataFrame(frows)
    fs.to_csv(res_dir / "female_share.csv", index=False)
    print("\n4. OVERALL FEMALE SHARE: first coder minus second, percentage points [95% bootstrap interval]")
    print(fs[["pair", "t2i_images", "images", "share_x", "share_y", "diff_pp", "ci_low", "ci_high"]].to_string(index=False))
    (res_dir / "female_share.tex").write_text(tex(ftex, [f"{s} images" for s, _ in subsets],
                                              "Difference in pooled female share (percentage points, 95\\% bootstrap interval).",
                                              "tab:female-share", first="First minus second"), encoding="utf-8")

    
    cb = {s["scenario_id"]: s for s in json.loads(args.codebook.read_text(encoding="utf-8"))}
    prows, unmatched = [], []
    for k in keys:
        a, b = main_recs[k], vol_by[k]
        if not (a.get("human_marks") and b.get("human_marks")):
            continue
        rt = flatten_results.role_table(cb[a["scenario_id"]])
        def people(rec):
            pos = {m["pid"]: m for m in rec["human_marks"]["people"]}
            return [dict(p, x=pos[p["pid"]]["x"], y=pos[p["pid"]]["y"]) for p in rec["response"]["people"] if p["pid"] in pos]
        pa, pb = people(a), people(b)
        pairs = match_people(pa, pb, args.match_distance)
        unmatched.append(dict(image_key=k, main_only=len(pa) - len(pairs), vol_only=len(pb) - len(pairs)))
        grp = lambda role: (lambda t: t["category"] if t["subtype"] in ("none", "") else f"{t['category']}:{t['subtype']}")(rt[role])
        for i, j, dist in pairs:
            p, q = pa[i], pb[j]
            prows.append(dict(image_key=k, scenario_id=a["scenario_id"], volunteer=b["coder_id"], pid_main=p["pid"],
                              pid_vol=q["pid"], distance=round(dist, 3), gender_main=p["gender"], gender_vol=q["gender"],
                              role_main=p["role"], role_vol=q["role"], category_main=rt[p["role"]]["category"],
                              category_vol=rt[q["role"]]["category"], group_main=grp(p["role"]), group_vol=grp(q["role"])))
    pm, um = pd.DataFrame(prows), pd.DataFrame(unmatched)
    if len(pm):
        pm.to_csv(res_dir / "person_matches.csv", index=False)
        conf = pd.crosstab(pm.category_main, pm.category_vol, margins=True, margins_name="total")
        conf.to_csv(res_dir / "person_confusion.csv")
        print(f"\n5. PERSON LEVEL, humans only: {len(pm)} people matched by marker position "
              f"(within {args.match_distance:.2f} of the image width); unmatched: {int(um.main_only.sum())} only in Main, "
              f"{int(um.vol_only.sum())} only in Vol")
        for lab, c in (("gender", "gender"), ("exact role", "role"), ("role category", "category"), ("category + subtype", "group")):
            print(f"   same {lab:<18} {kappa_str(pm[c + '_main'].astype(str).values, pm[c + '_vol'].astype(str).values)}")
        print("   Role category, Main (rows) vs Vol (columns):")
        print(conf.to_string())
        off = pm[pm.category_main != pm.category_vol].groupby(["category_main", "category_vol"]).size().sort_values(ascending=False)
        if len(off):
            print("   Most frequent category disagreements:\n   " +
                  "\n   ".join(f"{a} -> {b}: {n}" for (a, b), n in off.head(6).items()))

    vrows = []
    for v in sorted({r["coder_id"] for r in vol_recs}):
        ks = [k for k in keys if vol_by[k]["coder_id"] == v]
        if not ks:
            continue
        r = image_level(C["Main"], C["Vol"], ks)
        sub = pm[pm.volunteer == v] if len(pm) else pm
        vrows.append(dict(volunteer=v, images=len(ks), women_per_image=r["Women per image"], men_per_image=r["Men per image"],
                          women_per_category=r["Women per role category"], men_per_category=r["Men per role category"],
                          person_category=kappa_str(sub.category_main.astype(str).values, sub.category_vol.astype(str).values) if len(sub) else "--",
                          median_seconds=float(np.median([vol_by[k].get("time_spent") or 0 for k in ks]))))
    pv = pd.DataFrame(vrows)
    pv.to_csv(res_dir / "per_volunteer.csv", index=False)
    print("\n6a. EACH VOLUNTEER vs MAIN (non-crowded images; person_category = same category for matched people)")
    print(pv.to_string(index=False))
    srows = []
    for s in sorted(set(scen_of[keys])):
        ks = [k for k in keys if scen_of[k] == s]
        row = dict(scenario=s, images=len(ks))
        for c in ALL:
            i = C[c]["img"].loc[ks]
            row[f"W/M {c}"] = f"{int(i.n_female.sum())}/{int(i.n_male.sum())}"
        for x, y in PAIRS:
            ix, iy = C[x]["img"].loc[ks], C[y]["img"].loc[ks]
            row[f"same W {x}-{y}"] = f"{int((ix.n_female.values == iy.n_female.values).sum())}/{len(ks)}"
        srows.append(row)
    ps = pd.DataFrame(srows)
    ps.to_csv(res_dir / "per_scenario.csv", index=False)
    print(f"\n6b. PER SCENARIO ({', '.join(f'{k} = {name[k]}' for k in VLMS)}): women/men (W/M) counted by each coder, "
          "and images with the same number of women")
    print(ps.to_string(index=False))

  
    ck = sorted(crowded & coded)
    if ck:
        def maj(c, f):
            d = C[c]["img"].loc[ck]
            fe, ma = d[f"{f}_n_female"], d[f"{f}_n_male"]
            return np.select([fe + ma == 0, fe > ma, ma > fe], ["none or unclear", "mostly female", "mostly male"], "equal")
        cp = pd.DataFrame({"image_key": ck, "t2i_model": [t2i.get(k, "") for k in ck]})
        for f in ("foreground", "focal"):
            for c in ALL:
                cp[f"{f}_{name[c]}"] = maj(c, f)
        cp.to_csv(res_dir / "crowded_prominence.csv", index=False)
        print(f"\n7. CROWDED FALLBACK IMAGES ({len(ck)}): same majority gender of foreground / focal")
        for f in ("foreground", "focal"):
            parts = [f"{plabel(x, y)} {int((cp[f'{f}_{name[x]}'] == cp[f'{f}_{name[y]}']).sum())}/{len(ck)}" for x, y in PAIRS]
            print(f"   {f}: " + " | ".join(parts))


    mrows = []
    for x, y in PAIRS:
        for sub, ks in subsets[1:]:
            r = image_level(C[x], C[y], ks)
            mrows.append({"pair": plabel(x, y), "t2i_images": sub, "images": len(ks),
                          **{m: r[m] for m in ("People per image", "Women per image", "Men per image",
                                                "Women per role category", "Men per role category")}})
    bm = pd.DataFrame(mrows)
    bm.to_csv(res_dir / "by_image_model.csv", index=False)
    print("\n9. BY IMAGE MODEL: per-image agreement on each T2I model's images ('% identical / % within one')")
    print(bm.to_string(index=False))

    rows10 = []
    for s_ in sorted(set(scen_of[keys])):
        ks = [k for k in keys if scen_of[k] == s_]
        for g in groups:
            row = dict(scenario=s_, category=g)
            for c in ALL:
                f, m = group_arrays(C[c], ks, g)
                row[name[c]] = f"{int(f.sum())}/{int(f.sum() + m.sum())}"
            rows10.append(row)
    cps = pd.DataFrame(rows10)
    cps.to_csv(res_dir / "category_by_scenario.csv", index=False)
    show = cps[cps[[name[c] for c in ALL]].nunique(axis=1) > 1]
    show = show[show.category.isin(["RECEIVING", "ASSISTING", "PRIMARY", "OBSERVING:watching"])]
    print("\n10. ROLE CATEGORY BY SCENARIO (women/people), rows where the coders differ, main categories only "
          "(all rows in category_by_scenario.csv)")
    print(show.to_string(index=False) if len(show) else "   none")

    rows11 = []
    for c in ALL:
        i = C[c]["img"].loc[keys]
        r = C[c]["roles"][C[c]["roles"].image_key.isin(keys)]
        rows11.append(dict(coder=name[c], people=int(i.n_people_listed.sum()), unclear_gender=int(i.n_unclear.sum()),
                           baby_not_applicable=int(i.n_not_applicable.sum()),
                           unclear_role=int(r.loc[r.category == "UNCLEAR_ROLE", "n"].sum()),
                           none_of_the_above=int(r.loc[r.category == "OTHER_UNLISTED", "n"].sum()),
                           images_with_crowd_group=int((~i.exact_count).sum())))
    uc = pd.DataFrame(rows11)
    uc.to_csv(res_dir / "unclear_use.csv", index=False)
    print("\n11. USE OF 'UNCLEAR' AND FALLBACK OPTIONS, totals on these images")
    print(uc.to_string(index=False))

    qpath = out / "adjudication_queue.csv"
    im, iv = C["Main"]["img"].loc[keys], C["Vol"]["img"].loc[keys]
    hm = pd.DataFrame({"image_key": keys, "n_female_main": im.n_female.values, "n_male_main": im.n_male.values,
                       "n_female_vol": iv.n_female.values, "n_male_vol": iv.n_male.values})
    dis = hm[(hm.n_female_main != hm.n_female_vol) | (hm.n_male_main != hm.n_male_vol)]
    def listing(C_, k):
        p = C_["people"][C_["people"].image_key == k]
        return "; ".join(f"{r.pid}:{r.role_id}:{r.gender}@{str(r.where).split(', ')[-1]}" for r in p.itertuples())
    if tags:
        if not qpath.exists():
            print("\n8. Run once without subset options to create the adjudication queue.")
            print(f"\nAll outputs in {res_dir}")
            return
        print(f"\n8. Subset run: uses the adjudication decisions in {qpath}")
    elif qpath.exists() and pd.read_csv(qpath, dtype=str).fillna("")[["decided_female", "decided_male"]].ne("").any().any():
        print(f"\n8. {qpath.name} already has decisions; left unchanged.")
    else:
        q = dis.assign(volunteer=dis.image_key.map(lambda k: vol_by[k]["coder_id"]),
                       image_path=dis.image_key.map(lambda k: main_recs[k]["image_path"]),
                       main_people=dis.image_key.map(lambda k: listing(C["Main"], k)),
                       vol_people=dis.image_key.map(lambda k: listing(C["Vol"], k)),
                       decided_female="", decided_male="", reason="")
        q.to_csv(qpath, index=False, encoding="utf-8-sig")
        print(f"\n8. ADJUDICATION: {len(q)} of {len(keys)} non-crowded images differ between the humans in the number "
              f"of women or men -> {qpath}")
    q = pd.read_csv(qpath, dtype=str).fillna("")
    q = q[q.image_key.isin(keys)]
    if len(q) and (q[["decided_female", "decided_male"]] == "").any().any():
        print(f"   Gold comparison waits for decisions: {(q.decided_female != '').sum()} of {len(q)} decided.")
    else:
        gold = hm.rename(columns={"n_female_main": "n_female", "n_male_main": "n_male"})[["image_key", "n_female", "n_male"]].set_index("image_key")
        for _, r in q.iterrows():
            if r.image_key not in gold.index:
                continue
            gold.loc[r.image_key, ["n_female", "n_male"]] = [int(r.decided_female), int(r.decided_male)]
        grows, gtex = [], []
        for c in VLMS + ["Main", "Vol"]:
            cells = [name[c]]
            for sub, ks in subsets:
                i, g = C[c]["img"].loc[ks], gold.loc[ks]
                r = share_diff(i.n_female.to_numpy(float), i.n_male.to_numpy(float),
                               g.n_female.to_numpy(float), g.n_male.to_numpy(float), rng)
                grows.append(dict(coder=name[c], t2i_images=sub, women_per_image=fmt(i.n_female.values, g.n_female.values),
                                  men_per_image=fmt(i.n_male.values, g.n_male.values), **r))
                if sub == "all":
                    cells += [grows[-1]["women_per_image"], grows[-1]["men_per_image"]]
                cells.append(f"{r['diff_pp']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]")
            gtex.append(cells)
        gc = pd.DataFrame(grows)
        gc.to_csv(res_dir / "gold_counts.csv", index=False)
        print("\n8. AGAINST THE ADJUDICATED REFERENCE COUNTS (non-crowded)")
        print(gc[["coder", "t2i_images", "women_per_image", "men_per_image", "share_x", "share_y", "diff_pp", "ci_low", "ci_high"]]
              .to_string(index=False))
        (res_dir / "gold_counts.tex").write_text(tex(
            gtex, ["Women per image", "Men per image"] + [f"Female share: {s} images" for s, _ in subsets],
            "Each coder against the adjudicated reference counts (counts: \\% identical / \\% within one; "
            "female share: coder minus reference, percentage points, 95\\% bootstrap interval).", "tab:gold",
            first="Coder"), encoding="utf-8")
    print(f"\nAll outputs in {res_dir}")


if __name__ == "__main__":
    main()
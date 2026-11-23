"""
choose_threshold.py - choose n, the largest number of people per image that can still be coded reliably.

  python choose_threshold.py                     (run validate_humans.py once before, to build the volunteer file)
  python choose_threshold.py --human-bar 0.95 --vlm-margin 3     (stricter main rule, if decided in advance)

The script uses CODING AGREEMENT ONLY. It never computes a female share or any gender result, so the
threshold cannot be tuned to the findings.

Rule (main, fixed in advance; options change the numbers, the logic stays):
  Counts  for the images kept (size <= n), the two human coders give identical counts of women AND of men
          in >= --human-bar of images, and the primary VLM's agreement with the humans (average of VLM vs
          main coder and VLM vs volunteer) is within --vlm-margin points of the humans' agreement.
  Roles   for the same images, the primary VLM's role-category agreement with the humans is within
          --role-margin points of the humans' role-category agreement (no absolute bar: roles are partly
          ambiguous at any size, so the VLM is held to the human level).
  n       scanning n upwards from the smallest n that keeps --min-images images, n is the last value
          before the first n that fails; so every smaller threshold passes as well.

Sections (printed, and saved to vlm_results/threshold/):
  1 per-image agreement by size group, with 95% Wilson intervals              agreement_by_size_group.csv
  2 cumulative agreement for every candidate n, pass/fail, the chosen n        cumulative_by_n.csv
  3 smooth curve (logistic regression): size where agreement falls below the bar, bootstrap interval
  4 VLM vs VLM on ALL images (many more large scenes): does agreement drop at the same size?   vlm_vlm_by_n.csv
  5 sorting check: does the primary VLM put validation images on the same side of n as the humans?
  6 coverage on all images at n-2, n, n+2: images kept, by T2I model, cells losing over half, interval width
  7 sensitivity: the n each combination of bar and margins would give        sensitivity_grid.csv
  8 batch check: batch 1 (samples 1-10) vs batch 2 (samples 11-20), crowding and size by T2I model
  figure: agreement_vs_size.png / .pdf ;  decision record: threshold_decision.txt
"""

import argparse
import itertools
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from validate_humans import flat

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED, BOOT = 2026, 2000
GROUPS = [(1, 4), (5, 6), (7, 8), (9, 10), (11, 14), (15, 99)]



def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n (good for small samples)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pct(x):
    x = np.asarray(x, float)
    return 100 * x.mean() if len(x) else np.nan


def count_agree(X, Y, keys):
    a, b = X["img"].loc[keys], Y["img"].loc[keys]
    return ((a.n_female.values == b.n_female.values) & (a.n_male.values == b.n_male.values)).astype(float)


def role_agree(X, Y, keys):
    """1 if, for every role category, both coders placed the same number of women and of men in it."""
    def table(C):
        r = C["roles"][C["roles"].image_key.isin(keys)]
        return r.groupby(["image_key", "category"])[["n_female", "n_male"]].sum()
    m = table(X).join(table(Y), how="outer", lsuffix="_x", rsuffix="_y").fillna(0)
    m["same"] = (m.n_female_x == m.n_female_y) & (m.n_male_x == m.n_male_y)
    ok = m.groupby(level=0)["same"].all()
    return ok.reindex(keys, fill_value=True).values.astype(float)


def logistic(x, y, iters=50):
    """Logistic regression y ~ 1 + x by Newton's method; returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(2), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def crossing(b, bar):
    """Size at which the fitted agreement falls to the bar (None if it never does)."""
    if b[1] >= 0:
        return np.nan
    return (math.log(bar / (1 - bar)) - b[0]) / b[1]


def choose_n(cum, bar, margin, role_margin, min_images):
    """Largest n before the first failing n (only n with at least min_images kept images are judged)."""
    ok = cum[cum.images >= min_images]
    chosen = np.nan
    for _, r in ok.iterrows():
        passes = (r.hh_count >= 100 * bar) and (r.hh_count - r.vlm_count <= margin) and \
                 (r.hh_role - r.vlm_role <= role_margin)
        if not passes:
            break
        chosen = int(r.n)
    return chosen



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="vlm_results", type=Path)
    ap.add_argument("--main-coder", default="C01")
    ap.add_argument("--primary", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--secondary", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--selection", default="private/selection.csv", type=Path)
    ap.add_argument("--human-bar", type=float, default=0.90)
    ap.add_argument("--vlm-margin", type=float, default=5.0)
    ap.add_argument("--role-margin", type=float, default=5.0)
    ap.add_argument("--min-images", type=int, default=20, help="smallest number of kept images to judge an n")
    ap.add_argument("--n-max", type=int, default=25)
    args = ap.parse_args()
    out = args.results_dir / "threshold"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250)

    vol_file = args.results_dir / "human_validation" / "results_volunteers_human_vlm_v1_main.jsonl"
    if not vol_file.exists():
        raise SystemExit("Run validate_humans.py once first (it builds the volunteer file).")
    C = {"Main": flat(args.results_dir / f"results_{args.main_coder}_human_vlm_v1_main.jsonl"),
         "Vol": flat(vol_file), "P": flat(args.primary), "S": flat(args.secondary)}
    pname, sname = (str(C[k]["img"]["vlm_model_requested"].iloc[0]) for k in ("P", "S"))

    
    sel = pd.read_csv(args.selection)
    crowded = set(sel.loc[sel["image_type"] == "crowded_fallback", "image_key"]) if "image_type" in sel else set()
    keys = sorted((set(sel.image_key) - crowded) & set(C["Main"]["img"].index) & set(C["Vol"]["img"].index)
                  & set(C["P"]["img"].index))
    size = (C["Main"]["img"].loc[keys, "n_people_listed"].values + C["Vol"]["img"].loc[keys, "n_people_listed"].values) / 2
    hh_c, hh_r = count_agree(C["Main"], C["Vol"], keys), role_agree(C["Main"], C["Vol"], keys)
    vm_c, vv_c = count_agree(C["P"], C["Main"], keys), count_agree(C["P"], C["Vol"], keys)
    vm_r, vv_r = role_agree(C["P"], C["Main"], keys), role_agree(C["P"], C["Vol"], keys)
    vlm_c, vlm_r = (vm_c + vv_c) / 2, (vm_r + vv_r) / 2
    print(f"Validation: {len(keys)} non-crowded images | size = average of the two humans' people counts "
          f"(median {np.median(size):.1f}, max {size.max():.1f})")
    print(f"Primary VLM: {pname} | rule: humans >= {100 * args.human_bar:.0f}% identical women+men counts, "
          f"VLM within {args.vlm_margin:g} points; roles: VLM within {args.role_margin:g} points of humans")

    
    rows = []
    for lo, hi in GROUPS:
        m = (size >= lo) & (size <= hi + 0.5)
        n = int(m.sum())
        k = int(hh_c[m].sum())
        rows.append(dict(people=f"{lo}-{hi}" if hi < 99 else f"{lo}+", images=n,
                         hh_count=pct(hh_c[m]), hh_ci=("%.0f-%.0f" % wilson(k, n)) if n else "--",
                         vlm_count=pct(vlm_c[m]), hh_role=pct(hh_r[m]), vlm_role=pct(vlm_r[m])))
    g = pd.DataFrame(rows)
    g.to_csv(out / "agreement_by_size_group.csv", index=False)
    print("\n1. AGREEMENT BY SIZE GROUP (% of images; hh = human vs human, vlm = primary VLM vs humans, "
          "average of the two; 95% Wilson interval for humans)")
    print(g.round(1).to_string(index=False))

    rows = []
    for n in range(1, args.n_max + 1):
        m = size <= n + 0.5
        k = int(m.sum())
        if not k:
            continue
        lo, hi = wilson(int(hh_c[m].sum()), k)
        rows.append(dict(n=n, images=k, hh_count=pct(hh_c[m]), hh_low=lo, hh_high=hi, vlm_count=pct(vlm_c[m]),
                         hh_role=pct(hh_r[m]), vlm_role=pct(vlm_r[m])))
    cum = pd.DataFrame(rows)
    cum["count_gap"], cum["role_gap"] = cum.hh_count - cum.vlm_count, cum.hh_role - cum.vlm_role
    try:                                       # exact one-sided test: is human agreement on kept images above the bar?
        from scipy.stats import binomtest
        cum["p_above_bar"] = [round(binomtest(int(round(h * k / 100)), int(k), args.human_bar,
                                              alternative="greater").pvalue, 3) for h, k in zip(cum.hh_count, cum.images)]
    except ImportError:
        pass
    cum["passes"] = (cum.hh_count >= 100 * args.human_bar) & (cum.count_gap <= args.vlm_margin) & \
                    (cum.role_gap <= args.role_margin)
    n_star = choose_n(cum, args.human_bar, args.vlm_margin, args.role_margin, args.min_images)
    if not np.isnan(n_star):          # do not claim sizes no image had: shrink n to the largest size actually kept
        n_star = int(math.ceil(size[size <= n_star + 0.5].max() - 0.5))
    cum.to_csv(out / "cumulative_by_n.csv", index=False)
    print(f"\n2. KEEP IMAGES WITH AT MOST n PEOPLE: agreement on what is kept (gap = humans minus VLM; "
          f"n judged only when at least {args.min_images} images are kept)")
    print(cum.round(1).to_string(index=False))
    if "p_above_bar" in cum:
        conf = cum[(cum.images >= args.min_images) & (cum.p_above_bar < 0.05)]
        if len(conf):
            print(f"   p_above_bar = exact one-sided binomial test that human agreement exceeds {100 * args.human_bar:.0f}%; "
                  f"the sample CONFIRMS it (p < 0.05) up to n = {int(conf.n.max())}")
    if np.isnan(n_star):
        print("   NO n passes the rule. Report counts and roles with this limitation, or revisit the rule BEFORE "
              "any gender analysis and say so.")
    else:
        print(f"   CHOSEN n = {n_star}: the largest n before the first failure, shrunk to the largest image size "
              "actually present (no claim about sizes the sample never had)")

   
    def boot_cross(y):
        y = np.asarray(y, float)
        if y.min() == y.max():
            return np.nan, np.nan, np.nan
        est = crossing(logistic(size, y), args.human_bar)
        bs = []
        for _ in range(BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            bs.append(crossing(logistic(size[i], y[i]), args.human_bar))
        bs = np.array(bs, float)
        return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
    print(f"\n3. SMOOTH CURVE (logistic regression of agreement on size): size at which agreement on images of that "
          f"size falls to {100 * args.human_bar:.0f}% [95% bootstrap interval]")
    for lab, y in (("humans, counts", hh_c), (f"{pname} vs humans, counts", np.r_[vm_c, vv_c])):
        if lab.startswith(pname):
            sz = np.r_[size, size]
            est = crossing(logistic(sz, y), args.human_bar)
            bs = []
            for _ in range(BOOT):
                i = rng.integers(0, len(size), len(size))
                ii = np.r_[i, i + len(size)]
                if y[ii].min() != y[ii].max():
                    bs.append(crossing(logistic(sz[ii], y[ii]), args.human_bar))
            lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
        else:
            est, lo, hi = boot_cross(y)
        print(f"   {lab:<40} {est:5.1f} people [{lo:.1f}, {hi:.1f}]")
    print("   (this is agreement AT a size; section 2 is agreement on everything UP TO a size, which is what the rule uses)")

    
    both = C["P"]["img"].index.intersection(C["S"]["img"].index)
    pa, sa = C["P"]["img"].loc[both], C["S"]["img"].loc[both]
    nonc = pa.exact_count.values & sa.exact_count.values
    vsize = (pa.n_people_listed.values + sa.n_people_listed.values) / 2
    vv = ((pa.n_female.values == sa.n_female.values) & (pa.n_male.values == sa.n_male.values)).astype(float)
    rows = []
    for n in range(1, args.n_max + 1):
        m = nonc & (vsize <= n + 0.5)
        at = nonc & (np.abs(vsize - n) <= 0.5)
        rows.append(dict(n=n, images_kept=int(m.sum()), vlm_vlm_count_kept=pct(vv[m]),
                         images_at_size=int(at.sum()), vlm_vlm_count_at_size=pct(vv[at])))
    vvt = pd.DataFrame(rows)
    vvt.to_csv(out / "vlm_vlm_by_n.csv", index=False)
    print(f"\n4. {pname} vs {sname} ON ALL {len(both)} IMAGES ({int(nonc.sum())} non-crowded): identical women+men counts, "
          "on everything up to n and on images of about size n")
    print(vvt.round(1).to_string(index=False))

    
    if not np.isnan(n_star):
        human_small = size <= n_star + 0.5
        vlm_small = C["P"]["img"].loc[keys, "n_people_listed"].values <= n_star
        same = (human_small == vlm_small).mean() * 100
        print(f"\n5. SORTING CHECK at n = {n_star}: {pname} puts {same:.1f}% of validation images on the same side of n "
              f"as the humans ({int((human_small & ~vlm_small).sum())} that humans call small it calls large, "
              f"{int((~human_small & vlm_small).sum())} the other way)")

  
    allp = C["P"]["img"].loc[both].copy()
    allp["noncrowd"] = nonc
    print("\n6. COVERAGE ON ALL IMAGES (size judged by the primary VLM's count; only non-crowded images are eligible)")
    cov_rows = []
    cands = [n_star - 2, n_star, n_star + 2] if not np.isnan(n_star) else [8, 10, 12]
    for n in cands:
        kept = allp.noncrowd & (allp.n_people_listed <= n)
        cells = allp.assign(kept=kept).groupby(["scenario_id", "t2i_model"])["kept"].mean()
        widths = []
        for (s, t), _ in cells.items():
            d = allp[kept & (allp.scenario_id == s) & (allp.t2i_model == t)]
            f, mm = d.n_female.to_numpy(float), d.n_male.to_numpy(float)
            if len(d) < 3 or (f + mm).sum() == 0:
                continue
            idx = rng.integers(0, len(d), (500, len(d)))
            sh = f[idx].sum(1) / np.maximum((f[idx] + mm[idx]).sum(1), 1)
            widths.append(50 * (np.percentile(sh, 97.5) - np.percentile(sh, 2.5)))
        row = dict(n=n, kept=int(kept.sum()), of=len(allp), kept_pct=round(100 * kept.mean(), 1))
        for t in sorted(allp.t2i_model.unique()):
            row[f"kept_{t}"] = f"{int((kept & (allp.t2i_model == t)).sum())}/{int((allp.t2i_model == t).sum())}"
        row["cells_losing_over_half"] = int((cells < 0.5).sum())
        row["median_interval_halfwidth_pp"] = round(float(np.median(widths)), 1) if widths else np.nan
        cov_rows.append(row)
        if n == n_star:
            lost = cells[cells < 0.5]
            lost_list = ", ".join(f"{s} {t} ({v:.0%} kept)" for (s, t), v in lost.items()) or "none"
    cv = pd.DataFrame(cov_rows)
    cv.to_csv(out / "coverage.csv", index=False)
    print(cv.to_string(index=False))
    if not np.isnan(n_star):
        print(f"   Cells keeping under half their images at n = {n_star} (describe, do not quantify): {lost_list}")
    print("   median_interval_halfwidth_pp: typical +/- of a scenario x model female share with the images kept")

  
    grid = []
    for bar, mg in itertools.product([0.85, 0.90, 0.95], [3, 5, 7]):
        grid.append(dict(human_bar=f"{100 * bar:.0f}%", vlm_and_role_margin=mg,
                         n=choose_n(cum, bar, mg, mg, args.min_images)))
    sg = pd.DataFrame(grid).pivot(index="human_bar", columns="vlm_and_role_margin", values="n")
    sg.to_csv(out / "sensitivity_grid.csv")
    print("\n7. SENSITIVITY: n under other pass marks (rows: human bar; columns: VLM and role margin, points)")
    print(sg.to_string())

    allp["batch"] = np.where(allp["sample"].astype(int) <= 10, "1-10", "11-20")
    if allp.batch.nunique() > 1:
        b = allp.groupby(["t2i_model", "batch"]).agg(images=("noncrowd", "size"),
                                                     crowded_pct=("noncrowd", lambda x: round(100 * (1 - x.mean()), 1)),
                                                     mean_people=("n_people_listed", lambda x: round(x.mean(), 2)))
        b.to_csv(out / "batch_check.csv")
        print("\n8. BATCH CHECK (primary VLM): do samples 11-20 look like samples 1-10?")
        print(b.to_string())
    else:
        print("\n8. BATCH CHECK: only one batch coded so far; rerun after coding samples 11-20.")

    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xmax = int(min(args.n_max, math.ceil(max(size.max(), vsize[nonc].max() if nonc.any() else 0)) + 1))
        mids = {"1-4": 2.5, "5-6": 5.5, "7-8": 7.5, "9-10": 9.5, "11-14": 12.5, "15+": 16.0}
        gg = g.assign(x=g.people.map(mids)).dropna(subset=["x"])
        gg = gg[gg.images > 0]
        c = cum[(cum.images >= 5) & (cum.n <= xmax)]
        v = vvt[(vvt.n <= xmax)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        ax = axes[0]
        ax.plot(c.n, c.hh_count, "-", color="C0", label="Humans, all images up to n (rule)")
        ax.plot(c.n, c.vlm_count, "--", color="C1", label=f"{pname} vs humans, up to n")
        lo_hi = np.array([wilson(int(round(r.hh_count * r.images / 100)), int(r.images)) for r in gg.itertuples()])
        yerr = [np.clip(gg.hh_count.to_numpy(float) - lo_hi[:, 0], 0, None),
                np.clip(lo_hi[:, 1] - gg.hh_count.to_numpy(float), 0, None)]
        ax.errorbar(gg.x, gg.hh_count, yerr=yerr, fmt="o", color="C0",
                    capsize=3, label="Humans, images OF that size (group, 95% CI)")
        ax.plot(gg.x, gg.vlm_count, "s", color="C1", label=f"{pname} vs humans, images of that size")
        for r in gg.itertuples():                       
            ax.annotate(f"{int(r.images)} img", (r.x, r.hh_count), textcoords="offset points", xytext=(6, -10),
                        fontsize=7, color="C0")
        vv_ = v[v.images_at_size >= 10]
        ax.plot(vv_.n, vv_.vlm_vlm_count_at_size, ":", color="C2", label=f"{pname} vs {sname}, images of that size (all images)")
        ax.axhline(100 * args.human_bar, color="grey", lw=0.8)
        ax.set_title("Counts: identical women and men")
        ax.set_xlabel("People in image (n)")
        ax.set_ylabel("% of images where coders agree")
        ax = axes[1]
        ax.plot(c.n, c.hh_role, "-", color="C0", label="Humans, up to n")
        ax.plot(c.n, c.vlm_role, "--", color="C1", label=f"{pname} vs humans, up to n")
        ax.plot(gg.x, gg.hh_role, "o", color="C0", label="Humans, images of that size")
        ax.plot(gg.x, gg.vlm_role, "s", color="C1", label=f"{pname} vs humans, images of that size")
        for r in gg.itertuples():
            ax.annotate(f"{int(r.images)} img", (r.x, r.hh_role), textcoords="offset points", xytext=(6, -10),
                        fontsize=7, color="C0")
        ax.set_title("Roles: identical counts in every role category")
        ax.set_xlabel("People in image (n)")
        for a in axes:
            if not np.isnan(n_star):
                a.axvline(n_star, color="black", lw=0.8, ls="--")
            a.set_xlim(1, xmax)
            a.set_ylim(0, 105)
            a.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"agreement_vs_size.{ext}", dpi=200)
        print(f"\nFigure: {out / 'agreement_vs_size.png'} (left: counts, right: roles; lines = up to n, markers = of that size)")
    except Exception as e:
        print(f"(figure skipped: {e})")

    rec = dict(run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), chosen_n=None if np.isnan(n_star) else n_star,
               rule=dict(human_bar=args.human_bar, vlm_margin=args.vlm_margin, role_margin=args.role_margin,
                         min_images=args.min_images, primary_vlm=pname),
               validation_images=len(keys), all_images=int(len(both)))
    (out / "threshold_decision.txt").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"Decision record: {out / 'threshold_decision.txt'}")


if __name__ == "__main__":
    main()
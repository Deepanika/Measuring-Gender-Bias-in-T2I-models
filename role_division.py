"""
role_division.py - gender ROLE DIVISION in the images: within each scenario, are women a larger or smaller share of a
role category than they are of all the people in the scene?

  python role_division.py                                   main run (n = 13, no crowds, Gemini-3.8-flash coding)
  python role_division.py --n 7 --tag strict_n7             strict threshold
  python role_division.py --coder vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl --tag gpt_coder

This is NOT a TUS comparison (TUS has no roles). The reference is the scene itself:
  gap = female share within a role category - female share among all people (known gender) in the same images.
  gap > 0: women over-represented in that role; gap < 0: under-represented.
Categories: PRIMARY, DIRECTING, ASSISTING, RECEIVING, OTHER_TASK, OBSERVING (from the codebook join). People whose role
is unclear or "none of the above" count in the scene total but not in a category. Infants (gender not applicable) and
people of unclear gender are left out.

Same image selection as tus_compare.py: --n, --crowds, --exclude-scenarios, --models, --phrasings, --min-kept-share.
A category in a scenario x model is interpreted only with at least --min-people people (default 20).
A gap is a finding only if: Benjamini-Hochberg q < 0.05, its 95% interval excludes 0, and |gap| >= --min-diff (5 pp).
Intervals come from resampling images (paired: the category and the scene total are recomputed on the same draw).

OUTPUT  vlm_results/role_division/<tag>/
  cells.csv                 every scenario x model x category: women, men, shares, gap, interval, q, verdict
  pooled.csv / pooled.tex   all measured scenarios together, per T2I model and both: share and gap per category
                            (people-weighted) and the average of the per-scenario gaps (each scenario counts once)
  model_difference.csv      Gemini minus GPT images: difference in each category's pooled gap
  phrasing_difference.csv   use-case minus plain: difference in each category's pooled gap, per T2I model
  heatmap.png / .pdf        per-scenario gaps (rows: scenarios, columns: categories), one panel per T2I model
  pooled.png / .pdf         pooled gaps per category with intervals
  settings.json             the options used
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from validate_humans import flat

warnings.filterwarnings("ignore", category=RuntimeWarning)
BOOT, SEED = 2000, 2026
CATS = ["PRIMARY", "DIRECTING", "ASSISTING", "RECEIVING", "OTHER_TASK", "OBSERVING"]


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


def share(f, m):
    """female share in %, NaN where nobody is counted (works for numbers and arrays)"""
    f, m = np.asarray(f, float), np.asarray(m, float)
    t = f + m
    out = np.full(np.broadcast(f, t).shape, np.nan)
    np.divide(100 * f, t, out=out, where=t > 0)
    return out if out.shape else float(out)


def boot_p(d):
    d = d[~np.isnan(d)]
    if not len(d):
        return np.nan
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coder", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--crowd-file", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--crowds", choices=["exclude", "include"], default="exclude")
    ap.add_argument("--exclude-scenarios", default="")
    ap.add_argument("--models", default="both")
    ap.add_argument("--phrasings", default="both")
    ap.add_argument("--min-kept-share", type=float, default=0.5)
    ap.add_argument("--min-people", type=int, default=20)
    ap.add_argument("--min-diff", type=float, default=5.0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default="vlm_results/role_division", type=Path)
    args = ap.parse_args()
    out = args.out_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250)

    # ---- image selection (same rule as tus_compare.py)
    C = flat(args.coder)
    img, ppl = C["img"].copy(), C["people"]
    coder_name = str(img["vlm_model_requested"].iloc[0])
    img["crowded"] = ~img["exact_count"]
    if str(args.crowd_file) and Path(args.crowd_file).exists():
        other = flat(args.crowd_file)["img"]
        img = img.loc[img.index.intersection(other.index)]
        img["crowded"] = img["crowded"] | ~other.loc[img.index, "exact_count"]
    models = ["gemini", "openai"] if args.models == "both" else [m.strip() for m in args.models.split(",")]
    phr = ["plain", "use-case"] if args.phrasings == "both" else [p.strip() for p in args.phrasings.split(",")]
    drop = {s.strip() for s in args.exclude_scenarios.split(",") if s.strip()}
    base = img[img.t2i_model.isin(models) & img.phrasing.isin(phr) & ~img.scenario_id.isin(drop)].copy()
    keep = pd.Series(True, index=base.index)
    if args.crowds == "exclude":
        keep &= ~base.crowded
    if args.n > 0:
        keep &= base.n_people_listed <= args.n
    base["kept"] = keep
    print(f"Coder {coder_name} | kept {int(keep.sum())} of {len(base)} images (crowds {args.crowds}, "
          f"n {'no limit' if args.n <= 0 else '<= ' + str(args.n)}) | categories need >= {args.min_people} people per cell")

    # ---- per-image counts: women / men overall and per category
    p = ppl[ppl.image_key.isin(base.index[base.kept]) & ppl.gender.isin(["female", "male"])].copy()
    p["cat"] = p["category"].where(p["category"].isin(CATS), "OTHER")

    def arrays(keys):
        """dict: name -> (female array, male array) over images in keys, for 'ALL' and each category"""
        q = p[p.image_key.isin(keys)]
        res = {}
        g = q.groupby(["image_key", "gender"]).size().unstack(fill_value=0).reindex(keys, fill_value=0)
        res["ALL"] = (g.get("female", pd.Series(0, index=keys)).to_numpy(float),
                      g.get("male", pd.Series(0, index=keys)).to_numpy(float))
        for c in CATS:
            gc = q[q.cat == c].groupby(["image_key", "gender"]).size().unstack(fill_value=0).reindex(keys, fill_value=0)
            res[c] = (gc.get("female", pd.Series(0, index=keys)).to_numpy(float),
                      gc.get("male", pd.Series(0, index=keys)).to_numpy(float))
        return res

    def boot_sums(arr, n_img):
        idx = rng.integers(0, n_img, size=(BOOT, n_img))
        return {k: (f[idx].sum(1), m[idx].sum(1)) for k, (f, m) in arr.items()}

    # ---- per scenario x model
    order = sorted(base.scenario_id.unique(), key=lambda s: (int(s[1]), int(s[3])))
    rows, boots = [], {}
    for s in order:
        for m in models:
            cell = base[(base.scenario_id == s) & (base.t2i_model == m)]
            ks = list(cell.index[cell.kept])
            kept_share = len(ks) / len(cell) if len(cell) else 0
            measured = len(ks) > 0 and kept_share >= args.min_kept_share
            arr = arrays(ks) if ks else None
            bs = boot_sums(arr, len(ks)) if measured else None
            if measured:
                boots[(s, m)] = (arr, bs, ks)
            fa, ma = arr["ALL"] if arr else (np.zeros(0), np.zeros(0))
            all_share = float(share(fa.sum(), ma.sum())) if arr else np.nan
            for c in CATS:
                fc, mc = arr[c] if arr else (np.zeros(0), np.zeros(0))
                nf, nm = int(fc.sum()), int(mc.sum())
                row = dict(scenario=s, model=m, category=c, images_kept=len(ks), images_total=len(cell),
                           women=nf, men=nm, people=nf + nm, scene_female_share=round(all_share, 1),
                           category_female_share=round(float(share(nf, nm)), 1) if nf + nm else np.nan)
                if not measured:
                    row.update(status=f"described only ({100 * kept_share:.0f}% of images kept)" if ks else "no images kept")
                elif nf + nm < args.min_people:
                    row.update(status=f"too few people (< {args.min_people})")
                else:
                    gap_b = share(*bs[c]) - share(*bs["ALL"])
                    row.update(status="measured", gap_pp=round(float(share(nf, nm)) - all_share, 1),
                               ci_low=round(float(np.nanpercentile(gap_b, 2.5)), 1),
                               ci_high=round(float(np.nanpercentile(gap_b, 97.5)), 1), p=boot_p(gap_b))
                rows.append(row)
    cells = pd.DataFrame(rows)
    for col in ("gap_pp", "ci_low", "ci_high", "p"):
        if col not in cells:
            cells[col] = np.nan
    tested = cells.status == "measured"
    cells["q_bh"] = np.nan
    cells.loc[tested, "q_bh"] = bh(cells.loc[tested, "p"].values)
    sig = tested & (cells.q_bh < 0.05) & ((cells.ci_low > 0) | (cells.ci_high < 0)) & (cells.gap_pp.abs() >= args.min_diff)
    cells["finding"] = sig
    cells["verdict"] = np.select(
        [~tested, sig & (cells.gap_pp > 0), sig & (cells.gap_pp < 0), tested & (cells.gap_pp.abs() < args.min_diff)],
        [cells.status, "women over-represented", "women under-represented", f"negligible (< {args.min_diff:g} pp)"],
        "no clear difference")
    cells.to_csv(out / "cells.csv", index=False)

    show = cells[tested].copy()
    show["q_bh"] = show.q_bh.round(3)
    print("\nPER SCENARIO x MODEL (categories with enough people): female share in the scene vs in the category, gap in pp")
    print(show[["scenario", "model", "category", "images_kept", "women", "men", "scene_female_share",
                "category_female_share", "gap_pp", "ci_low", "ci_high", "q_bh", "verdict"]].to_string(index=False))
    f = cells[cells.finding]
    print(f"\n{int(tested.sum())} scenario x model x category gaps tested; {len(f)} findings "
          f"(q < 0.05, interval excludes 0, |gap| >= {args.min_diff:g} pp):")
    for v, g in f.groupby("verdict"):
        print(f"   {v}: " + ", ".join(f"{r.scenario} {r.model} {r.category} ({r.category_female_share} vs {r.scene_female_share})"
                                     for r in g.itertuples()))

    # ---- pooled over measured scenarios (bootstrap within each cell, then add up)
    def pooled_for(cells_keys):
        if not cells_keys:
            return None
        tot = {k: [0.0, 0.0] for k in ["ALL"] + CATS}
        btot = {k: [np.zeros(BOOT), np.zeros(BOOT)] for k in ["ALL"] + CATS}
        for key in cells_keys:
            arr, bs, _ = boots[key]
            for k in tot:
                tot[k][0] += arr[k][0].sum()
                tot[k][1] += arr[k][1].sum()
                btot[k][0] += bs[k][0]
                btot[k][1] += bs[k][1]
        return tot, btot

    prow = []
    groups = [(m, [k for k in boots if k[1] == m]) for m in models] + ([("both", list(boots))] if len(models) > 1 else [])
    pooled_store = {}
    for label, keys in groups:
        pr = pooled_for(keys)
        if pr is None:
            continue
        tot, btot = pr
        pooled_store[label] = pr
        scene = float(share(*tot["ALL"]))
        for c in CATS:
            fc, mc = tot[c]
            per = cells[(cells.category == c) & tested & cells.model.isin([label] if label != "both" else models)]
            gap_b = share(*btot[c]) - share(*btot["ALL"])
            prow.append(dict(model=label, category=c, scenarios=len(keys), women=int(fc), men=int(mc),
                             scene_female_share=round(scene, 1),
                             category_female_share=round(float(share(fc, mc)), 1) if fc + mc else np.nan,
                             gap_pp=round(float(share(fc, mc)) - scene, 1) if fc + mc else np.nan,
                             ci_low=round(float(np.nanpercentile(gap_b, 2.5)), 1),
                             ci_high=round(float(np.nanpercentile(gap_b, 97.5)), 1), p=boot_p(gap_b),
                             mean_scenario_gap=round(per.gap_pp.mean(), 1) if len(per) else np.nan,
                             scenarios_with_enough_people=len(per)))
    pooled = pd.DataFrame(prow)
    pooled["q_bh"] = bh(pooled.p.values)
    pooled["finding"] = (pooled.q_bh < 0.05) & ((pooled.ci_low > 0) | (pooled.ci_high < 0)) & (pooled.gap_pp.abs() >= args.min_diff)
    pooled.to_csv(out / "pooled.csv", index=False)
    print("\nPOOLED over measured scenarios (people-weighted; mean_scenario_gap = average of per-scenario gaps, "
          "each scenario counted once):")
    print(pooled.drop(columns=["p"]).assign(q_bh=pooled.q_bh.round(3)).to_string(index=False))

    # ---- model and phrasing differences in pooled gaps
    def gap_boot(keys, filt=None):
        """pooled gap per category for the given cells; optional image filter (phrasing) -> (point, boot)"""
        tot = {k: [0.0, 0.0] for k in ["ALL"] + CATS}
        btot = {k: [np.zeros(BOOT), np.zeros(BOOT)] for k in ["ALL"] + CATS}
        for key in keys:
            arr, _, ks = boots[key]
            sel = np.array([filt(k) for k in ks]) if filt else np.ones(len(ks), bool)
            if sel.sum() < 2:
                continue
            idx = rng.integers(0, int(sel.sum()), size=(BOOT, int(sel.sum())))
            for k in tot:
                f, m = arr[k][0][sel], arr[k][1][sel]
                tot[k][0] += f.sum()
                tot[k][1] += m.sum()
                btot[k][0] += f[idx].sum(1)
                btot[k][1] += m[idx].sum(1)
        pt = {c: float(share(*tot[c]) - share(*tot["ALL"])) for c in CATS}
        bt = {c: share(*btot[c]) - share(*btot["ALL"]) for c in CATS}
        return pt, bt

    if len(models) == 2:
        common = sorted({k[0] for k in boots if k[1] == "gemini"} & {k[0] for k in boots if k[1] == "openai"})
        pg, bg = gap_boot([(s, "gemini") for s in common])
        po, bo = gap_boot([(s, "openai") for s in common])
        md = pd.DataFrame([dict(category=c, scenarios=len(common), gap_gemini=round(pg[c], 1), gap_openai=round(po[c], 1),
                                difference=round(pg[c] - po[c], 1),
                                ci_low=round(float(np.nanpercentile(bg[c] - bo[c], 2.5)), 1),
                                ci_high=round(float(np.nanpercentile(bg[c] - bo[c], 97.5)), 1)) for c in CATS])
        md["differs"] = (md.ci_low > 0) | (md.ci_high < 0)
        md.to_csv(out / "model_difference.csv", index=False)
        print(f"\nGEMINI minus GPT images: difference in pooled gap, same {len(common)} scenarios measured for both")
        print(md.to_string(index=False))
    if len(phr) == 2:
        ph = []
        for m in models:
            keys = [k for k in boots if k[1] == m]
            pu, bu = gap_boot(keys, lambda k: base.at[k, "phrasing"] == "use-case")
            pp_, bp = gap_boot(keys, lambda k: base.at[k, "phrasing"] == "plain")
            for c in CATS:
                d = bu[c] - bp[c]
                ph.append(dict(model=m, category=c, gap_usecase=round(pu[c], 1), gap_plain=round(pp_[c], 1),
                               difference=round(pu[c] - pp_[c], 1), ci_low=round(float(np.nanpercentile(d, 2.5)), 1),
                               ci_high=round(float(np.nanpercentile(d, 97.5)), 1)))
        phd = pd.DataFrame(ph)
        phd["differs"] = (phd.ci_low > 0) | (phd.ci_high < 0)
        phd.to_csv(out / "phrasing_difference.csv", index=False)
        print("\nUSE-CASE minus PLAIN: difference in pooled gap per T2I model")
        print(phd.to_string(index=False))

    # ---- LaTeX: pooled table
    lines = []
    for r in pooled.itertuples():
        star = "$^{*}$" if r.finding else ""
        lines.append(f"{r.model} & {r.category.replace('_', ' ').title()} & {r.women} / {r.men} & {r.category_female_share:.1f} & "
                     f"{r.scene_female_share:.1f} & {r.gap_pp:+.1f}{star} [{r.ci_low:+.1f}, {r.ci_high:+.1f}] & "
                     f"{r.mean_scenario_gap:+.1f} ({r.scenarios_with_enough_people}) \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrcccc@{}}\n"
           "\\toprule\nImages & Role category & Women / men & Female \\% in role & Female \\% in scene & "
           "Gap (pp) [95\\% CI] & Mean scenario gap (n) \\\\\n\\midrule\n" + "\n".join(lines) +
           "\n\\bottomrule\n\\end{tabular}}\n\\caption{Gender role division: female share within each role category vs "
           f"among all people in the same images. $^{{*}}$ q $<$ 0.05, interval excludes 0, $|$gap$|$ $\\geq$ {args.min_diff:g} pp. "
           f"Settings: {args.tag}. [Write your caption here.]}}\n\\label{{tab:role-division}}\n\\end{{table}}\n")
    (out / "pooled.tex").write_text(tex, encoding="utf-8")

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 0.3 * len(order) + 1.6), squeeze=False)
        for ax, m in zip(axes[0], models):
            sub = cells[cells.model == m]
            mat = np.full((len(order), len(CATS)), np.nan)
            for r in sub[sub.status == "measured"].itertuples():
                mat[order.index(r.scenario), CATS.index(r.category)] = r.gap_pp
            im = ax.imshow(mat, cmap="RdBu", vmin=-50, vmax=50, aspect="auto")
            for r in sub[sub.finding].itertuples():
                ax.plot(CATS.index(r.category), order.index(r.scenario), "k.", ms=5)
            ax.set_xticks(range(len(CATS)))
            ax.set_xticklabels([c.replace("_", "\n").title() for c in CATS], fontsize=7)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=7)
            ax.set_title(f"{m} images", fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="female share in role minus in scene (pp)")
        fig.suptitle("Role division per scenario (blank = not measured or < %d people; dot = finding)" % args.min_people,
                     fontsize=9)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"heatmap.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        labels = [l for l in ["gemini", "openai", "both"] if l in pooled.model.unique()]
        w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            g = pooled[pooled.model == l].set_index("category").reindex(CATS)
            x = np.arange(len(CATS)) + (i - (len(labels) - 1) / 2) * w
            err = [np.clip(g.gap_pp - g.ci_low, 0, None), np.clip(g.ci_high - g.gap_pp, 0, None)]
            ax.bar(x, g.gap_pp, w, yerr=err, capsize=2, label=f"{l} images")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels([c.replace("_", " ").title() for c in CATS], fontsize=8)
        ax.set_ylabel("female share in role minus in scene (pp)")
        ax.legend(fontsize=8)
        ax.set_title(f"Pooled role division ({args.tag})", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"pooled.{ext}", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"(figures skipped: {e})")

    (out / "settings.json").write_text(json.dumps(dict(
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), coder=coder_name, n=args.n, crowds=args.crowds,
        exclude_scenarios=sorted(drop), models=models, phrasings=phr, min_kept_share=args.min_kept_share,
        min_people=args.min_people, min_diff=args.min_diff, images_kept=int(base.kept.sum())), indent=1), encoding="utf-8")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
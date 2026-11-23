"""
tus_compare.py - compare the female share among people doing each scenario's activity in the images with the
TUS 2024 expected share, at 3-digit (MAIN), 2-digit and 1-digit level.

  python tus_compare.py --mapping tus_role_mapping_by_level_1.xlsx                      main run (n = 13, no crowds)
  python tus_compare.py --mapping tus_role_mapping_by_level_1.xlsx --n 7 --tag strict
  python tus_compare.py --mapping ... --crowds include --n 0 --tag all_images            every image
  python tus_compare.py --mapping ... --exclude-scenarios D5S1,D7S2 --models gemini --tag gemini_only
  python tus_compare.py --mapping ... --coder vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl --tag gpt_coder
  python tus_compare.py --mapping ... --sensitivity --tag sensitivity_roles

WHICH IMAGES
  --n N              keep images whose coder listed at most N people (0 = no limit). Default 13.
  --crowds           exclude (default): drop images where EITHER VLM used a crowd group (the "non-crowded" rule)
                     include: keep them; then only individually listed people are counted (crowd groups have no
                     exact count), and their listed number is a lower bound, so --n is checked against it.
  --exclude-scenarios  comma-separated scenario ids to leave out
  --models           openai, gemini or both (default both): which image generator's images to analyse
  --phrasings        plain, use-case or both (default both)
  --min-kept-share   a scenario x model is measured only if at least this share of its images is kept
                     (default 0.5); otherwise it is listed as "described only" with no test.

WHAT IS COMPARED (from the mapping workbook)
  Role inclusion sheet: which roles count at each level ("count at 3-digit" etc.).
  Scenario x level sheet: TUS female and male rates; BONUS columns: a second comparison inside a scenario
  (e.g. customers in D1S2 vs TUS shopping). --sensitivity adds the "sensitivity: roles to add" roles.
  Expected female share = w f / (w f + (1 - w) m), w = female share of persons aged 6+ (Population weight sheet).
  Rounding range: Table 6 rates have one decimal, so the expected share is also given as the range for
  rates +/- 0.05; a difference counts only if the observed 95% interval lies entirely outside that range.

OUTPUT  vlm_results/tus_compare/<tag>/
  results.csv / results_main.tex    every scenario x level x model (+ bonus rows): images, women, men, observed
                                    share with 95% bootstrap interval (resampling images), expected share and range,
                                    difference, p, Benjamini-Hochberg q (3-digit rows only), direction
  model_difference.csv              Gemini minus GPT, per scenario, at each level (when both models are kept)
  phrasing_difference.csv           use-case minus plain, per scenario x model, at each level
  figure_3digit.png / .pdf          difference from TUS at 3-digit level with intervals and rounding ranges
  settings.json                     the exact options used (for the report and for reproducibility)
"""

import argparse
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from validate_humans import flat

warnings.filterwarnings("ignore", category=RuntimeWarning)
BOOT, SEED = 2000, 2026
LEVELS = ["3-digit", "2-digit", "1-digit"]


# ------------------------------------------------------------------ mapping
def read_mapping(path):
    sx = pd.read_excel(path, "Scenario x level")
    sx = sx[sx["scenario_id"].astype(str).str.fullmatch(r"D\dS\d")].copy()
    ri = pd.read_excel(path, "Role inclusion")
    pw = pd.read_excel(path, "Population weight", header=None)
    w = 0.4967
    for i in range(len(pw)):
        for j in range(pw.shape[1] - 1):
            if isinstance(pw.iat[i, j], str) and pw.iat[i, j].strip().lower().startswith("w used"):
                try:
                    w = float(pw.iat[i, j + 1])
                except (TypeError, ValueError):
                    pass
    return sx, ri, w


def roles_at(ri, sid, level):
    col = f"count at {level}"
    r = ri[(ri.scenario_id == sid) & (ri[col].astype(str).str.upper() == "Y")]
    return list(r.role_id)


def split_roles(v):
    return [x.strip() for x in str(v).split(";") if x.strip() and x.strip().lower() != "nan"]


def expected(f, m, w):
    if f is None or m is None or (f + m) == 0:
        return np.nan
    return 100 * w * f / (w * f + (1 - w) * m)


def rounding_range(f, m, w):
    lf, hf, lm, hm = max(f - 0.05, 0), f + 0.05, max(m - 0.05, 0), m + 0.05
    lo = 100 * w * lf / (w * lf + (1 - w) * hm) if (lf + hm) > 0 else 0.0
    hi = 100 * w * hf / (w * hf + (1 - w) * lm) if (hf + lm) > 0 else 100.0
    return lo, hi


# ------------------------------------------------------------------ statistics
def boot_shares(fem, mal, rng):
    """pooled female share and its bootstrap distribution, resampling images"""
    n = len(fem)
    tot = fem.sum() + mal.sum()
    if n == 0 or tot == 0:
        return np.nan, np.array([np.nan])
    idx = rng.integers(0, n, size=(BOOT, n))
    f, t = fem[idx].sum(1), (fem[idx] + mal[idx]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(t > 0, 100 * f / t, np.nan)
    return 100 * fem.sum() / tot, b


def test_against_range(obs, boot, lo, hi):
    """p-value for 'true share lies inside [lo, hi]'; 1 if the observed share is inside"""
    b = boot[~np.isnan(boot)]
    if np.isnan(obs) or not len(b):
        return np.nan
    if obs > hi:
        return min(1.0, 2 * (np.sum(b <= hi) + 1) / (len(b) + 1))
    if obs < lo:
        return min(1.0, 2 * (np.sum(b >= lo) + 1) / (len(b) + 1))
    return 1.0


def bh(p):
    """Benjamini-Hochberg q-values"""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        ranked = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def direction(obs, exp, significant):
    if not significant or np.isnan(obs) or np.isnan(exp):
        return "consistent with TUS" if not np.isnan(obs) else ""
    if abs(exp - 50) < 0.5:
        return "departs from parity toward " + ("women" if obs > 50 else "men")
    if (obs - 50) * (exp - 50) < 0:
        return "reverses"
    return "amplifies" if abs(obs - 50) > abs(exp - 50) else "reduces"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True, type=Path)
    ap.add_argument("--coder", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path,
                    help="VLM results whose people and roles are counted (primary coder by default)")
    ap.add_argument("--crowd-file", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path,
                    help="second VLM used for the non-crowded rule (both VLMs list everyone); '' to use the coder only")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--crowds", choices=["exclude", "include"], default="exclude")
    ap.add_argument("--exclude-scenarios", default="")
    ap.add_argument("--models", default="both", help="openai, gemini or both")
    ap.add_argument("--phrasings", default="both", help="plain, use-case or both")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--min-kept-share", type=float, default=0.5)
    ap.add_argument("--sensitivity", action="store_true", help="add the 'sensitivity: roles to add' roles")
    ap.add_argument("--min-diff", type=float, default=5.0,
                    help="smallest difference (percentage points) treated as meaningful; smaller ones are 'negligible'")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default="vlm_results/tus_compare", type=Path)
    args = ap.parse_args()
    out = args.out_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250)

    sx, ri, w = read_mapping(args.mapping)
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    C = flat(args.coder)
    img, ppl = C["img"].copy(), C["people"]
    coder_name = str(img["vlm_model_requested"].iloc[0])

    # ---- image selection
    img["crowded"] = ~img["exact_count"]
    if args.crowd_file and str(args.crowd_file) and Path(args.crowd_file).exists():
        other = flat(args.crowd_file)["img"]
        img = img.loc[img.index.intersection(other.index)]
        img["crowded"] = img["crowded"] | ~other.loc[img.index, "exact_count"]
    models = ["gemini", "openai"] if args.models == "both" else [m.strip() for m in args.models.split(",")]
    phr = ["plain", "use-case"] if args.phrasings == "both" else [p.strip() for p in args.phrasings.split(",")]
    drop_s = {s.strip() for s in args.exclude_scenarios.split(",") if s.strip()}
    base = img[img.t2i_model.isin(models) & img.phrasing.isin(phr) & ~img.scenario_id.isin(drop_s)]
    keep = pd.Series(True, index=base.index)
    if args.crowds == "exclude":
        keep &= ~base.crowded
    if args.n > 0:
        keep &= base.n_people_listed <= args.n
    base = base.assign(kept=keep)
    kept = base[base.kept]
    print(f"Coder: {coder_name} | images considered {len(base)} (models {models}, phrasings {phr}, "
          f"excluded scenarios {sorted(drop_s) or 'none'})")
    print(f"Kept: {len(kept)} (crowds {args.crowds}, n {'no limit' if args.n <= 0 else '<= ' + str(args.n)}) | "
          f"population weight w = {w:.4f}")
    if args.crowds == "include":
        print("Note: crowded images included; only individually listed people are counted (crowd groups have no count).")

    # people per image, gender female/male only
    p = ppl[ppl.image_key.isin(kept.index) & ppl.gender.isin(["female", "male"])]

    def counts(keys, roles):
        q = p[p.image_key.isin(keys) & p.role_id.isin(roles)]
        g = q.groupby(["image_key", "gender"]).size().unstack(fill_value=0).reindex(keys, fill_value=0)
        fem = g["female"].to_numpy(float) if "female" in g else np.zeros(len(keys))
        mal = g["male"].to_numpy(float) if "male" in g else np.zeros(len(keys))
        return fem, mal

    rows, cache = [], {}
    for _, r in sx.iterrows():
        sid, level = r.scenario_id, r.level
        if level not in levels or sid in drop_s:
            continue
        comparisons = [("main", roles_at(ri, sid, level) +
                        (split_roles(r.get("sensitivity: roles to add")) if args.sensitivity else []),
                        r.get("comparison code"), r.get("female rate (%)"), r.get("male rate (%)"))]
        if pd.notna(r.get("BONUS: roles compared")):
            comparisons.append(("bonus", split_roles(r["BONUS: roles compared"]), r.get("BONUS: TUS code"),
                                r.get("BONUS: female rate (%)"), r.get("BONUS: male rate (%)")))
        for kind, roles, code, f, m in comparisons:
            f, m = float(f), float(m)
            exp = expected(f, m, w)
            lo, hi = rounding_range(f, m, w)
            for model in models:
                cell = base[(base.scenario_id == sid) & (base.t2i_model == model)]
                ks = list(cell.index[cell.kept])
                share_kept = len(ks) / len(cell) if len(cell) else 0
                fem, mal = counts(ks, roles)
                obs, boot = boot_shares(fem, mal, rng)
                cache[(sid, level, kind, model)] = (ks, roles)
                status = ("no images kept" if not ks else
                          "measured" if share_kept >= args.min_kept_share and fem.sum() + mal.sum() > 0 else
                          "no participants coded" if share_kept >= args.min_kept_share else
                          f"described only ({100 * share_kept:.0f}% of images kept)")
                if np.isnan(exp):
                    status = "no TUS baseline (rates round to 0.0)" if status == "measured" else status
                measured = status == "measured"
                ci_lo, ci_hi = (np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)) if measured else (np.nan, np.nan)
                pval = test_against_range(obs, boot, lo, hi) if measured else np.nan
                sig = measured and (ci_hi < lo or ci_lo > hi)
                rows.append(dict(scenario=sid, level=level, comparison=kind, tus_code=code, model=model,
                                 images_kept=len(ks), images_total=len(cell), status=status,
                                 women=int(fem.sum()), men=int(mal.sum()),
                                 observed=round(obs, 1) if measured else np.nan,
                                 ci_low=round(ci_lo, 1) if measured else np.nan,
                                 ci_high=round(ci_hi, 1) if measured else np.nan,
                                 expected=round(exp, 1) if not np.isnan(exp) else np.nan,
                                 exp_low=round(lo, 1), exp_high=round(hi, 1),
                                 diff_pp=round(obs - exp, 1) if measured else np.nan,
                                 diff_ci=f"{ci_lo - exp:+.1f} to {ci_hi - exp:+.1f}" if measured else "",
                                 p=pval, outside_range=sig, roles="; ".join(roles),
                                 baseline_note=("rare activity (both TUS rates under 1%): read with the 2-digit row"
                                                if max(f, m) < 1.0 else "")))
    res = pd.DataFrame(rows)
    main_rows = (res.level == "3-digit") & (res.comparison == "main") & res.p.notna()
    res["q_bh"] = np.nan
    res.loc[main_rows, "q_bh"] = bh(res.loc[main_rows, "p"].values)
    big = res.diff_pp.abs() >= args.min_diff
    res["significant_main"] = main_rows & (res.q_bh < 0.05) & res.outside_range & big
    # direction: on MAIN rows only after the correction; on other rows by the interval (robustness)
    decisive = np.where(main_rows, res.significant_main, res.outside_range & big)
    res["direction"] = [direction(o, e, s) for o, e, s in zip(res.observed, res.expected, decisive)]
    small = res.observed.notna() & ~big
    res.loc[small, "direction"] = f"negligible (< {args.min_diff:g} pp)"
    res.to_csv(out / "results.csv", index=False)

    show = ["scenario", "level", "comparison", "tus_code", "model", "images_kept", "women", "men", "observed",
            "ci_low", "ci_high", "expected", "exp_low", "exp_high", "diff_pp", "q_bh", "direction", "status", "baseline_note"]
    for level in levels:
        t = res[res.level == level][show].copy()
        t["q_bh"] = t.q_bh.round(3)
        tag = " (MAIN; q = Benjamini-Hochberg over these rows)" if level == "3-digit" else " (robustness)"
        print(f"\n===== {level}{tag}: female share (%) among people doing the activity vs TUS =====")
        print(t.drop(columns="level").to_string(index=False))
    m = res[(res.level == "3-digit") & (res.comparison == "main") & (res.status == "measured")]
    print(f"\nMAIN (3-digit): {len(m)} scenario x model cells measured; {int(res.significant_main.sum())} differ from TUS "
          f"after Benjamini-Hochberg (q < 0.05), with the interval outside the rounding range and |difference| >= {args.min_diff:g} pp:")
    for d_, g in res[res.significant_main].groupby("direction"):
        print(f"   {d_}: " + ", ".join(f"{r.scenario} {r.model} ({r.observed} vs {r.expected})" for r in g.itertuples()))

    # ---- model and phrasing differences
    def diff_rows(a_keys, b_keys, roles):
        fa, ma = counts(a_keys, roles)
        fb, mb = counts(b_keys, roles)
        oa, ba = boot_shares(fa, ma, rng)
        ob, bb = boot_shares(fb, mb, rng)
        d = ba - bb
        return oa - ob, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

    md, pdiff = [], []
    measured = res[res.status == "measured"]
    for (sid, level, kind), g in measured.groupby(["scenario", "level", "comparison"]):
        if len(models) == 2 and set(g.model) == {"gemini", "openai"}:
            (kg, roles), (ko, _) = cache[(sid, level, kind, "gemini")], cache[(sid, level, kind, "openai")]
            dd, lo_, hi_ = diff_rows(kg, ko, roles)
            md.append(dict(scenario=sid, level=level, comparison=kind, gemini_minus_openai=round(dd, 1),
                           ci_low=round(lo_, 1), ci_high=round(hi_, 1), differs=(lo_ > 0 or hi_ < 0)))
        if len(phr) == 2:
            for model in g.model:
                ks, roles = cache[(sid, level, kind, model)]
                pu = [k for k in ks if base.at[k, "phrasing"] == "use-case"]
                pp = [k for k in ks if base.at[k, "phrasing"] == "plain"]
                if len(pu) >= 3 and len(pp) >= 3:
                    dd, lo_, hi_ = diff_rows(pu, pp, roles)
                    pdiff.append(dict(scenario=sid, level=level, comparison=kind, model=model,
                                      usecase_minus_plain=round(dd, 1), ci_low=round(lo_, 1), ci_high=round(hi_, 1),
                                      differs=(lo_ > 0 or hi_ < 0)))
    if md:
        mdf = pd.DataFrame(md)
        mdf.to_csv(out / "model_difference.csv", index=False)
        x = mdf[(mdf.level == "3-digit")]
        print(f"\nGEMINI minus GPT images (3-digit, percentage points): {int(x.differs.sum())} of {len(x)} intervals exclude 0")
        print(x.drop(columns="level").to_string(index=False))
    if pdiff:
        pdf = pd.DataFrame(pdiff)
        pdf.to_csv(out / "phrasing_difference.csv", index=False)
        x = pdf[(pdf.level == "3-digit")]
        print(f"\nUSE-CASE minus PLAIN (3-digit): {int(x.differs.sum())} of {len(x)} intervals exclude 0")
        print(x[x.differs].drop(columns="level").to_string(index=False) if x.differs.any() else "   none")

    # ---- LaTeX (main level)
    t = res[(res.level == "3-digit")].copy()
    lines = []
    for r in t.itertuples():
        name = r.scenario + (" (bonus)" if r.comparison == "bonus" else "")
        if r.status != "measured":
            lines.append(f"{name} & {r.tus_code:g} & {r.model} & {r.images_kept} & \\multicolumn{{4}}{{l}}{{{r.status}}} \\\\")
            continue
        star = "$^{*}$" if r.significant_main else ""
        lines.append(f"{name} & {r.tus_code:g} & {r.model} & {r.images_kept} & {r.observed:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}] & "
                     f"{r.expected:.1f} [{r.exp_low:.1f}, {r.exp_high:.1f}] & {r.diff_pp:+.1f}{star} & {r.direction} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{@{}lllrcccl@{}}\n\\toprule\n"
           "Scenario & TUS code & Images & Kept & Observed \\% [95\\% CI] & Expected \\% [rounding range] & Diff. (pp) & Direction \\\\\n"
           "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n"
           f"\\caption{{Female share among people doing each activity vs TUS 2024 (3-digit). $^{{*}}$ Benjamini--Hochberg q $<$ 0.05 "
           f"and interval outside the rounding range and |difference| $\\geq$ {args.min_diff:g} pp. Settings: {args.tag}. "
           "[Write your caption here.]}\n"
           "\\label{tab:tus-3digit}\n\\end{table}\n")
    (out / "results_main.tex").write_text(tex, encoding="utf-8")

    # ---- figure (3-digit main comparisons)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f3 = res[(res.level == "3-digit") & (res.comparison == "main")]
        scen = sorted(f3.scenario.unique(), key=lambda s: (int(s[1]), int(s[3])), reverse=True)
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(scen) + 1.2))
        ymap = {s: i for i, s in enumerate(scen)}
        for s in scen:
            e = f3[f3.scenario == s].iloc[0]
            ax.plot([e.exp_low - e.expected, e.exp_high - e.expected], [ymap[s]] * 2, color="0.8", lw=6, solid_capstyle="butt")
        for k, (model, mk) in enumerate([("gemini", "o"), ("openai", "s")]):
            g = f3[(f3.model == model) & (f3.status == "measured")]
            y = g.scenario.map(ymap) + (0.15 if k == 0 else -0.15)
            err = [np.clip(g.diff_pp - (g.ci_low - g.expected), 0, None), np.clip((g.ci_high - g.expected) - g.diff_pp, 0, None)]
            ax.errorbar(g.diff_pp, y, xerr=err,
                        fmt=mk, ms=4, capsize=2, label=f"{model} images")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(scen)))
        ax.set_yticklabels(scen)
        ax.set_xlabel("Observed minus TUS expected female share (percentage points)")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"3-digit comparison ({args.tag}); grey = TUS rounding range", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"figure_3digit.{ext}", dpi=200)
    except Exception as e:                       # the figure is optional; never lose the tables because of it
        print(f"(figure skipped: {e})")

    (out / "settings.json").write_text(json.dumps(dict(
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), mapping=str(args.mapping), coder=coder_name,
        crowd_file=str(args.crowd_file), n=args.n, crowds=args.crowds, exclude_scenarios=sorted(drop_s), models=models,
        phrasings=phr, levels=levels, min_kept_share=args.min_kept_share, sensitivity=args.sensitivity, w=w,
        min_diff=args.min_diff,
        images_considered=int(len(base)), images_kept=int(len(kept))), indent=1), encoding="utf-8")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
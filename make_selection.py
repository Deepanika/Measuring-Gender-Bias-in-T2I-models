"""
make_selection.py - draw the human-validation sample and assign coders (run ONCE, before coding starts).

  python make_selection.py vlm_results/results_gpt-6.1-sol_low_vlm_v1_main_images.csv ^
                           vlm_results/results_gemini-3.8-flash_low_vlm_v1_main_images.csv

SAMPLE   One image per scenario x T2I model x phrasing (27 x 2 x 2 = 108), drawn at random (fixed seed):
         from the cell's non-crowded images (both VLMs coded it OK and listed everyone individually);
         only if the cell has none, from its crowded images (both VLMs coded it OK). Each pick is labelled
         non_crowded or crowded_fallback, so the validation keeps the two apart.
CODERS   The main coder (C01) codes every image.
         Volunteers come from private/volunteers.csv (coder, max_images, crowd). Scenarios that contain a
         crowded pick ("crowd scenarios") go only to volunteers with crowd = yes. Volunteers get WHOLE
         scenarios (4 images each) up to their max_images, so they read as few codebooks as possible;
         scenarios are dealt heaviest first to whoever has the most time left relative to what they agreed to.
         If someone's number is not a multiple of 4, they are topped up with single images at the end.
OUTPUT   private/selection.csv      image_key, coders (+ information columns) for build_annotation_site.py
         private/cell_coverage.csv  every cell: images, non-crowded, what was picked and its type
After coding starts, change assignments by editing the coders column of private/selection.csv and
rebuilding; do not rerun this script, which would reassign scenarios.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

MIN_NONCROWD_BASE, MIN_PER_PERSON = 1.5, 0.4      # time model: non-crowded image = 1.5 min + 0.4 min per person
MIN_CROWDED = 12.0                                 # crowded image
MIN_CODEBOOK, MIN_START = 3.0, 12.0                # reading one scenario's roles; consent + rules


def load(path):
    d = pd.read_csv(path)
    d["exact_count"] = d["exact_count"].astype(str).str.lower().eq("true")
    return d.set_index("image_key")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vlm_a_images_csv")
    ap.add_argument("vlm_b_images_csv")
    ap.add_argument("--volunteers-file", default="private/volunteers.csv", type=Path)
    ap.add_argument("--main-coder", default="C01")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="private", type=Path)
    args = ap.parse_args()

    out = args.out_dir / "selection.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists. After coding has started, edit it by hand instead of re-drawing.\n"
                         f"Delete it first only if no coder has started yet.")

    # ---- images and the non-crowded flag
    a, b = load(args.vlm_a_images_csv), load(args.vlm_b_images_csv)
    keys = a.index.intersection(b.index)
    d = a.loc[keys, ["t2i_model", "scenario_id", "phrasing", "n_people_listed"]].copy()
    d["both_ok"] = (a.loc[keys, "run_status"].eq("ok") & b.loc[keys, "run_status"].eq("ok")).values
    d["noncrowd"] = d["both_ok"] & a.loc[keys, "exact_count"].values & b.loc[keys, "exact_count"].values
    d = d.reset_index()

    # ---- one image per cell
    rng = random.Random(args.seed)
    picked, cov = [], []
    for s in sorted(d.scenario_id.unique()):
        for m in sorted(d.t2i_model.unique()):
            for ph in ("plain", "use-case"):
                cell = d[(d.scenario_id == s) & (d.t2i_model == m) & (d.phrasing == ph)]
                nc = sorted(cell.loc[cell.noncrowd, "image_key"])
                cr = sorted(cell.loc[cell.both_ok & ~cell.noncrowd, "image_key"])
                kind, key = ("non_crowded", rng.choice(nc)) if nc else \
                            (("crowded_fallback", rng.choice(cr)) if cr else ("none", ""))
                cov.append(dict(scenario_id=s, t2i_model=m, phrasing=ph, images=len(cell), non_crowded=len(nc),
                                picked=key, image_type=kind))
                if key:
                    people = int(cell.set_index("image_key").loc[key, "n_people_listed"])
                    mins = MIN_CROWDED if kind == "crowded_fallback" else MIN_NONCROWD_BASE + MIN_PER_PERSON * people
                    picked.append(dict(image_key=key, scenario_id=s, t2i_model=m, phrasing=ph, image_type=kind,
                                       n_people_vlm=people, est_min=mins))
    sel, cov = pd.DataFrame(picked), pd.DataFrame(cov)
    crowd_scen = sorted(sel.loc[sel.image_type == "crowded_fallback", "scenario_id"].unique())

    # ---- volunteers
    vol = pd.read_csv(args.volunteers_file, dtype=str).fillna("")
    vol["max_images"] = vol["max_images"].astype(int)
    vol["crowd"] = vol["crowd"].str.strip().str.lower().isin(["yes", "y", "true", "1"])
    cap = dict(zip(vol.coder, vol.max_images))
    crowd_ok = set(vol.loc[vol.crowd, "coder"])
    left, mins, owner = dict(cap), {c: 0.0 for c in cap}, {}

    def deal(scenarios, allowed):
        scen_min = sel.groupby("scenario_id")["est_min"].sum() + MIN_CODEBOOK
        scen_n = sel.groupby("scenario_id").size()
        for s in sorted(scenarios, key=lambda x: -scen_min[x]):
            fits = [c for c in allowed if left[c] >= scen_n[s]]
            if not fits:
                continue
            c = min(fits, key=lambda x: ((mins[x] + scen_min[s]) / cap[x], x))   # most time left, relatively
            owner[s] = c
            left[c] -= scen_n[s]
            mins[c] += scen_min[s]

    deal(crowd_scen, sorted(crowd_ok))
    normal = [s for s in sel.scenario_id.unique() if s not in crowd_scen]
    deal(normal, sorted(cap))                              # crowd volunteers with room left can take these too
    sel["volunteer"] = sel["scenario_id"].map(owner).fillna("")

    # top up volunteers whose number is not a multiple of 4 with single images from unassigned scenarios
    for i in sel.index[sel.volunteer == ""]:
        s, crowded = sel.at[i, "scenario_id"], sel.at[i, "scenario_id"] in crowd_scen
        fits = [c for c in sorted(cap) if left[c] > 0 and (not crowded or c in crowd_ok)]
        if fits:
            c = max(fits, key=lambda x: left[x])
            sel.at[i, "volunteer"] = c
            left[c] -= 1
            mins[c] += sel.at[i, "est_min"]

    sel["coders"] = [f"{args.main_coder};{v}" if v else args.main_coder for v in sel.volunteer]
    cols = ["image_key", "coders", "scenario_id", "t2i_model", "phrasing", "image_type", "n_people_vlm"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sel[cols].to_csv(out, index=False)
    cov.to_csv(args.out_dir / "cell_coverage.csv", index=False)

    # ---- report
    n_cr = int((sel.image_type == "crowded_fallback").sum())
    print(f"Non-crowded images: {int(d.noncrowd.sum())} of {len(d)}")
    print(f"Sample: {len(sel)} images ({len(sel) - n_cr} non-crowded, {n_cr} crowded fallbacks) in "
          f"{(cov.image_type != 'none').sum()} of {len(cov)} cells")
    print(f"Crowd scenarios (contain a crowded pick, only for crowd volunteers): {', '.join(crowd_scen) or 'none'}")
    if (cov.image_type == "none").any():
        print("Cells with no usable image at all:\n" + cov[cov.image_type == "none"].to_string(index=False))
    rows = [dict(coder=args.main_coder, agreed="all", images=len(sel), crowded=n_cr,
                 codebooks=sel.scenario_id.nunique(),
                 est_minutes=round(sel.est_min.sum() + MIN_CODEBOOK * sel.scenario_id.nunique()))]
    for c in sorted(cap):
        mine = sel[sel.volunteer == c]
        rows.append(dict(coder=c, agreed=cap[c], images=len(mine), crowded=int((mine.image_type == "crowded_fallback").sum()),
                         codebooks=mine.scenario_id.nunique(),
                         est_minutes=round(mins[c] + MIN_START) if len(mine) else 0))
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("(volunteer minutes include ~12 min for consent and rules)\n")
    for c in sorted(cap):
        mine = sel[sel.volunteer == c]
        print(f"  {c}: {', '.join(sorted(mine.scenario_id.unique())) or '(nothing)'}")
    single = sel[sel.volunteer == ""]
    if len(single):
        print(f"\nWARNING: {len(single)} images have no volunteer (only {args.main_coder} codes them). "
              f"Recruit more, or accept them as single-coded:")
        print(single[["image_key", "scenario_id", "image_type"]].to_string(index=False))
    unused = {c: n for c, n in left.items() if n > 0}
    if unused:
        print(f"\nUnused agreed images: {unused}")
    print(f"\nWritten: {out} and {args.out_dir / 'cell_coverage.csv'}")


if __name__ == "__main__":
    main()

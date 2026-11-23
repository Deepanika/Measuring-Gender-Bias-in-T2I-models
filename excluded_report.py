"""
excluded_report.py - descriptive results for the images the main analysis leaves out (crowded, or more than n
people), plus the one scene question in the codebook (D8S2: gender of the cricket players on the TV screen).

  python excluded_report.py                 (n = 13, Gemini-3.8-flash coding, GPT-6.1-sol for the crowd rule)

Everything here is DESCRIPTIVE (shares, not tests). Crowd groups have only 8 crowded validation images behind them;
the scene question was answered identically by the coders on both validation images where it applied.

OUTPUT  vlm_results/excluded/
  excluded_by_cell.csv      per scenario x T2I model: images, excluded, of which crowded / larger than n
  crowd_mix.csv / .tex      crowd groups (primary coder) by T2I model and role category: share labelled
                            all/mostly male, mixed, all/mostly female, unclear
  scene_question.csv        D8S2 answers per T2I model, for each VLM, all images and kept images; VLM agreement
"""

import argparse
from pathlib import Path

import pandas as pd

from validate_humans import flat

MALE = {"all_male", "mostly_male"}
FEMALE = {"all_female", "mostly_female"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coder", default="vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--other", default="vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl", type=Path)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--out-dir", default="vlm_results/excluded", type=Path)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    A, B = flat(args.coder), flat(args.other)
    a, b = A["img"], B["img"]
    keys = a.index.intersection(b.index)
    d = a.loc[keys, ["scenario_id", "t2i_model", "phrasing", "n_people_listed"]].copy()
    d["crowded"] = ~(a.loc[keys, "exact_count"] & b.loc[keys, "exact_count"])
    d["large"] = ~d.crowded & (d.n_people_listed > args.n)
    d["excluded"] = d.crowded | d.large
    name_a = str(a["vlm_model_requested"].iloc[0])

    
    by_model = d.groupby("t2i_model").agg(images=("excluded", "size"), excluded=("excluded", "sum"),
                                          crowded=("crowded", "sum"), larger_than_n=("large", "sum"))
    by_model["crowded_pct"] = (100 * by_model.crowded / by_model.images).round(1)
    print(f"IMAGES LEFT OUT of the main analysis (crowded by either VLM, or more than {args.n} people by {name_a}):")
    print(by_model.to_string())
    cells = d.groupby(["scenario_id", "t2i_model"]).agg(images=("excluded", "size"), excluded=("excluded", "sum"),
                                                        crowded=("crowded", "sum"), larger_than_n=("large", "sum"))
    cells["excluded_pct"] = (100 * cells.excluded / cells.images).round(0)
    cells = cells.reset_index()
    cells.to_csv(out / "excluded_by_cell.csv", index=False)
    print("\nCells losing more than half their images:")
    print(cells[cells.excluded_pct > 50].to_string(index=False))

    
    cr = A.get("crowds")
    cpath = Path(f"{Path(args.coder).with_suffix('')}_crowds.csv")
    cr = pd.read_csv(cpath) if cr is None and cpath.exists() else cr
    if cr is not None and len(cr):
        cr = cr[cr.image_key.isin(d.index[d.crowded])].copy()
        cr["lean"] = cr.gender_mix.map(lambda g: "male-leaning" if g in MALE else "female-leaning" if g in FEMALE
                                       else "mixed" if g == "mixed" else "unclear")
        rows = []
        for (m, c), g in list(cr.groupby(["t2i_model", "category"])) + [((m, "ALL"), g) for m, g in cr.groupby("t2i_model")]:
            vc = g.lean.value_counts(normalize=True) * 100
            rows.append(dict(t2i_model=m, category=c, crowd_groups=len(g), images=g.image_key.nunique(),
                             **{k: round(vc.get(k, 0.0), 1) for k in ["male-leaning", "mixed", "female-leaning", "unclear"]}))
        mix = pd.DataFrame(rows).sort_values(["t2i_model", "category"])
        mix.to_csv(out / "crowd_mix.csv", index=False)
        print(f"\nCROWD GROUPS in crowded images, coded by {name_a} (% of groups; not validated against humans):")
        print(mix.to_string(index=False))
        bys = cr.groupby(["scenario_id", "t2i_model"]).lean.value_counts(normalize=True).unstack(fill_value=0).mul(100).round(0)
        bys["groups"] = cr.groupby(["scenario_id", "t2i_model"]).size()
        bys.to_csv(out / "crowd_mix_by_scenario.csv")
        print("\nCrowd-group gender lean by scenario (% of groups):")
        print(bys.to_string())
        lines = [f"{r.t2i_model} & {str(r.category).replace('_', ' ').title()} & {r.crowd_groups} & {r['male-leaning']:.0f} & "
                 f"{r['mixed']:.0f} & {r['female-leaning']:.0f} & {r['unclear']:.0f} \\\\" for _, r in mix.iterrows()]
        (out / "crowd_mix.tex").write_text(
            "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
            "Images & Role category & Groups & Male-leaning \\% & Mixed \\% & Female-leaning \\% & Unclear \\% \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Gender mix of crowd groups in crowded images "
            "(primary VLM; descriptive, not validated against humans). [Write your caption here.]}\n"
            "\\label{tab:crowd-mix}\n\\end{table}\n", encoding="utf-8")

   
    q = [c for c in a.columns if c.startswith("scene_q") and not c.endswith("_question")]
    if q:
        rows = []
        for col in q:
            sub = d[(d.scenario_id == "D8S2")].index
            for label, src in ((name_a, a), (str(b["vlm_model_requested"].iloc[0]), b)):
                for subset, ks in (("all images", sub), ("kept images", [k for k in sub if not d.at[k, "excluded"]])):
                    for m in sorted(d.loc[sub, "t2i_model"].unique()):
                        kk = [k for k in ks if d.at[k, "t2i_model"] == m]
                        vc = src.loc[kk, col].value_counts()
                        rows.append(dict(question=col, coder=label, images=subset, t2i_model=m, n=len(kk),
                                         **{str(k): int(v) for k, v in vc.items()}))
            both = [k for k in sub if pd.notna(a.at[k, col]) and pd.notna(b.at[k, col])]
            same = sum(str(a.at[k, col]) == str(b.at[k, col]) for k in both)
            print(f"\nSCENE QUESTION {col} (D8S2, players on the TV screen): the two VLMs give the same answer in "
                  f"{same}/{len(both)} images")
        sq = pd.DataFrame(rows).fillna(0)
        sq.to_csv(out / "scene_question.csv", index=False)
        print(sq.to_string(index=False))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
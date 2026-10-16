"""
vlm_agreement_basic.py - agreement between two VLM coders, per image, as one LaTeX table.

Usage:
    python vlm_agreement_basic.py vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl \
                                  vlm_results/results_gemini-3.8-flash_low_vlm_v1_main.jsonl

Only images coded successfully by both coders are compared, in three groups:
    All images | No crowd (both coders listed everyone) | Crowd (either coder reported a crowd)

Measures
    Answers (same category?)  -> % identical and Cohen's kappa
        crowd reported; majority gender of the foreground group; majority gender of the focal group
        (majority gender = mostly female / mostly male / equal / none or unclear; groups can be any size)
    Counts (same number?)     -> % identical and % within one person
        people, women, men per image; size of the foreground and focal groups;
        women and men per role type in each image (only image-role type pairs where either coder coded someone)

Output: the table printed in the terminal and saved as agreement_table.tex next to the first results file.
"""



import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GENDERS = ("female", "male", "unclear")


# ---------------------------------------------------------------- loading ---
def first_value(df, column, default=""):
    """First non-empty value of a column, or a default if the column is missing or empty."""
    if column not in df:
        return default
    values = df[column].dropna()
    return values.iloc[0] if len(values) else default


def effort_from_name(name):
    """Older results files have no effort column; the effort is still in the file name (..._low_vlm_...)."""
    m = re.search(r"_(minimal|low|medium|high)_vlm_", name)
    return m.group(1) if m else ""


def load(results_jsonl):
    """Read the tables flatten_results.py wrote next to a results file."""
    stem = Path(results_jsonl).with_suffix("")
    images_csv, roles_csv = Path(f"{stem}_images.csv"), Path(f"{stem}_role_counts.csv")
    if not images_csv.exists() or not roles_csv.exists():
        raise SystemExit(f"Run `python flatten_results.py {results_jsonl}` first.")
    images, roles = pd.read_csv(images_csv), pd.read_csv(roles_csv)
    images["exact_count"] = images["exact_count"].astype(str).str.lower().eq("true")
    model = first_value(images, "vlm_model_requested", stem.name)
    effort = first_value(images, "reasoning_effort") or effort_from_name(stem.name)
    return images, roles, f"{model} ({effort})" if effort else model


# --------------------------------------------------------------- measures ---
def pct_identical(a, b):
    return 100 * np.mean(np.asarray(a) == np.asarray(b))


def pct_within_one(a, b):
    return 100 * np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) <= 1)


def cohen_kappa(a, b):
    """Agreement corrected for chance: (observed - expected) / (1 - expected)."""
    a, b = np.asarray(a, dtype=str), np.asarray(b, dtype=str)
    observed = np.mean(a == b)
    expected = sum(np.mean(a == c) * np.mean(b == c) for c in set(a) | set(b))
    return np.nan if expected == 1 else (observed - expected) / (1 - expected)


def majority_gender(df, field, coder):
    """Majority gender of a foreground/focal group of any size (infants are not counted)."""
    f, m = df[f"{field}_n_female_{coder}"], df[f"{field}_n_male_{coder}"]
    return np.select([f + m == 0, f > m, m > f], ["none or unclear", "mostly female", "mostly male"], "equal")


# ------------------------------------------------------------------ table ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_a")
    ap.add_argument("results_b")
    args = ap.parse_args()
    img_a, roles_a, name_a = load(args.results_a)
    img_b, roles_b, name_b = load(args.results_b)

    # images coded successfully by both coders
    ok_a = set(img_a.loc[img_a.run_status == "ok", "image_key"])
    ok_b = set(img_b.loc[img_b.run_status == "ok", "image_key"])
    both = ok_a & ok_b
    print(f"Coder A: {name_a}\nCoder B: {name_b}\n"
          f"Coded OK: A {len(ok_a)}, B {len(ok_b)}, both {len(both)} "
          f"(of {len(set(img_a.image_key) | set(img_b.image_key))} images)")

    cols = ["image_key", "exact_count", "n_people_listed", "n_female", "n_male"] + \
        [f"{f}_{x}" for f in ("foreground", "focal") for x in ("n_female", "n_male", "n_unclear")]
    m = pd.merge(img_a.loc[img_a.image_key.isin(both), cols], img_b.loc[img_b.image_key.isin(both), cols],
                 on="image_key", suffixes=("_A", "_B"))
    for c in "AB":
        m[f"crowd_{c}"] = ~m[f"exact_count_{c}"]
        for f in ("foreground", "focal"):
            m[f"{f}_majority_{c}"] = majority_gender(m, f, c)
            m[f"{f}_size_{c}"] = sum(m[f"{f}_n_{g}_{c}"] for g in GENDERS)
    m["group"] = np.where(m.crowd_A | m.crowd_B, "Crowd", "No crowd")

    # women and men per role type per image, from both coders
    keep = ["image_key", "role_id", "category", "n_female", "n_male"]
    cells = pd.merge(roles_a.loc[roles_a.image_key.isin(both), keep],
                     roles_b.loc[roles_b.image_key.isin(both), keep].drop(columns="category"),
                     on=["image_key", "role_id"], how="outer", suffixes=("_A", "_B"))
    cells["category"] = cells["category"].fillna("unknown")
    cells = cells.fillna(0).groupby(["image_key", "category"], as_index=False)[
        ["n_female_A", "n_female_B", "n_male_A", "n_male_B"]].sum()
    cells = cells.merge(m[["image_key", "group"]], on="image_key")

    groups = {"All images": m.index == m.index, "No crowd": m.group == "No crowd", "Crowd": m.group == "Crowd"}

    def answer_row(label, var, all_only=False):
        out = [label]
        for g, mask in groups.items():
            sub = m[mask]
            if (all_only and g != "All images") or sub.empty:
                out.append("--")
                continue
            k = cohen_kappa(sub[f"{var}_A"], sub[f"{var}_B"])
            out.append(f"{pct_identical(sub[f'{var}_A'], sub[f'{var}_B']):.1f} "
                       f"({'--' if np.isnan(k) else f'{k:.2f}'})")
        return out

    def count_row(label, df, var, cell_mask=None):
        out = [label]
        for g in groups:
            sub = df if g == "All images" else df[df.group == g]
            if cell_mask is not None:
                sub = sub[(sub[f"{var}_A"] > 0) | (sub[f"{var}_B"] > 0)]
            out.append("--" if sub.empty else
                       f"{pct_identical(sub[f'{var}_A'], sub[f'{var}_B']):.1f} / "
                       f"{pct_within_one(sub[f'{var}_A'], sub[f'{var}_B']):.1f}")
        return out

    answers = [answer_row("Crowd reported", "crowd", all_only=True),
               answer_row("Foreground: majority gender", "foreground_majority"),
               answer_row("Focal: majority gender", "focal_majority")]
    counts = [count_row("People per image", m, "n_people_listed"),
              count_row("Women per image", m, "n_female"),
              count_row("Men per image", m, "n_male"),
              count_row("Foreground: number of people", m, "foreground_size"),
              count_row("Focal: number of people", m, "focal_size"),
              count_row("Women per role type", cells, "n_female", cell_mask=True),
              count_row("Men per role type", cells, "n_male", cell_mask=True)]
    n = {g: int(np.sum(mask)) for g, mask in groups.items()}

    # terminal
    head = ["", f"All (n={n['All images']})", f"No crowd (n={n['No crowd']})", f"Crowd (n={n['Crowd']})"]
    print("\n% identical (kappa)\n" + pd.DataFrame(answers, columns=head).to_string(index=False))
    print("\n% identical / % within one\n" + pd.DataFrame(counts, columns=head).to_string(index=False))

    # LaTeX
    def tex_rows(rows):
        return "\n".join(" & ".join(r) + r" \\" for r in rows)

    tex = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
 & \textbf{{All}} & \textbf{{No crowd}} & \textbf{{Crowd}} \\
 & ($n={n['All images']}$) & ($n={n['No crowd']}$) & ($n={n['Crowd']}$) \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\emph{{Same answer: \% identical ($\kappa$)}}}} \\
{tex_rows(answers)}
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\emph{{Same count: \% identical / \% within one person}}}} \\
{tex_rows(counts)}
\bottomrule
\end{{tabular}}
\caption{{Agreement between the two VLM coders, per image. [Write your caption here.]}}
\label{{tab:vlm-agreement}}
\end{{table}}
"""
    out = Path(args.results_a).parent / "agreement_table.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX table written to {out}")


if __name__ == "__main__":
    main()
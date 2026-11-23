"""Inter-VLM agreement per role category on the full set, split All / No crowd / Crowd.
Same method as vlm_agreement_basic.py (roles merged by role_id, summed per image x group,
cells where either coder placed someone of that gender)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from vlm_agreement_basic import load, pct_identical, pct_within_one

ORDER = ["PRIMARY", "DIRECTING", "ASSISTING", "RECEIVING", "OTHER_TASK:paid_work", "OTHER_TASK:domestic_chore",
         "OTHER_TASK:care", "OBSERVING:watching", "OBSERVING:own_activity", "UNCLEAR_ROLE", "OTHER_UNLISTED"]
a, b = sys.argv[1], sys.argv[2]
img_a, roles_a, _ = load(a)
img_b, roles_b, _ = load(b)
both = set(img_a.loc[img_a.run_status == "ok", "image_key"]) & set(img_b.loc[img_b.run_status == "ok", "image_key"])
ex = pd.merge(img_a[["image_key", "exact_count"]], img_b[["image_key", "exact_count"]], on="image_key")
ex = ex[ex.image_key.isin(both)]
ex["grp"] = np.where(~ex.exact_count_x | ~ex.exact_count_y, "Crowd", "No crowd")
keep = ["image_key", "role_id", "category", "subtype", "n_female", "n_male"]
c = pd.merge(roles_a.loc[roles_a.image_key.isin(both), keep],
             roles_b.loc[roles_b.image_key.isin(both), keep].drop(columns=["category", "subtype"]),
             on=["image_key", "role_id"], how="outer", suffixes=("_A", "_B"))
c["category"] = c["category"].fillna("unknown")
c["subtype"] = c["subtype"].fillna("none")
c["g"] = [k if s in ("none", "") else f"{k}:{s}" for k, s in zip(c.category, c.subtype)]
num = ["n_female_A", "n_female_B", "n_male_A", "n_male_B"]
c[num] = c[num].fillna(0)
c = c.merge(ex[["image_key", "grp"]], on="image_key")
n = {"All": len(ex), "No crowd": int((ex.grp == "No crowd").sum()), "Crowd": int((ex.grp == "Crowd").sum())}

def cell(df, v):
    s = df[(df[v + "_A"] > 0) | (df[v + "_B"] > 0)]
    return (len(s), "--" if s.empty else f"{pct_identical(s[v+'_A'], s[v+'_B']):.1f} / {pct_within_one(s[v+'_A'], s[v+'_B']):.1f}")

pc = c.groupby(["image_key", "category", "grp"], as_index=False)[num].sum()
for v in ("n_female", "n_male"):
    print("CHECK pooled", v, [cell(pc if g == "All" else pc[pc.grp == g], v) for g in n])

gc = c.groupby(["image_key", "g", "grp"], as_index=False)[num].sum()
rows = []
for g in ORDER:
    r = [g]
    for v in ("n_female", "n_male"):
        for grp in n:
            d = gc[gc.g == g] if grp == "All" else gc[(gc.g == g) & (gc.grp == grp)]
            k, s = cell(d, v)
            r.append(s); r.append(k)
    rows.append(r)
cols = ["group"] + [f"{v}_{grp}_{x}" for v in ("women", "men") for grp in n for x in ("agree", "n")]
out = pd.DataFrame(rows, columns=cols)
print(n)
print(out.to_string(index=False))
out.to_csv("vlm_results/category_agreement_vlm_full.csv", index=False)

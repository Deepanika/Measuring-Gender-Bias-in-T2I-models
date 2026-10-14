"""
flatten_results.py - turn raw VLM results (JSONL) into analysis tables.

  python flatten_results.py vlm_results/results_gpt-6.1-sol_vlm_v1_main.jsonl

Writes next to the input file:
  <stem>_images.csv       one row per image   (counts, foreground, focal, scene answers, tokens, cost)
  <stem>_people.csv       one row per person  (role, category, subtype, gender)
  <stem>_role_counts.csv  one row per image x role, including zero rows (the main analysis table)
  <stem>_crowds.csv       one row per crowd group (role, where, gender mix label, size band)
  <stem>_human_crowd_queue.csv  every crowd group, with blank columns for a human coder (images with crowds)
Raw JSONL is never modified; re-run this script whenever the codebook joins change.
Prices (USD per 1M tokens) are looked up per model in MODEL_PRICES (check them on the providers' pricing
pages before reporting); PRICE_IN / PRICE_OUT / PRICE_CACHED in .env override them for every model.
Cached input is charged at the full input price unless PRICE_CACHED is set (a conservative estimate).
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from vlm_prompt import FALLBACK_ROLES, PROMINENCE_FIELDS, scene_question_ids

load_dotenv()
CODEBOOK_PATH = Path(os.getenv("CODEBOOK_PATH", "codebook_final.json"))
MODEL_PRICES = {                      # (input, output) USD per 1M tokens, standard (non-batch) rates
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),  # Google's listed rate through 31 Dec 2026 ($1.50 / $7.50 from 2027)
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def prices(model):
    p_in, p_out = MODEL_PRICES.get(model or "", (None, None))
    p_in = float(os.getenv("PRICE_IN", p_in if p_in is not None else 0))
    p_out = float(os.getenv("PRICE_OUT", p_out if p_out is not None else 0))
    p_cached = float(os.getenv("PRICE_CACHED", p_in))
    return p_in, p_cached, p_out
META = ["image_key", "t2i_model", "dimension", "within_dimension", "scenario_id", "phrasing", "sample",
        "vlm_provider", "vlm_model_requested", "vlm_model_returned", "reasoning_effort", "vlm_prompt_version",
        "codebook_version", "run_tag", "mode"]


def role_table(scenario):
    rows = [dict(role_id=r["role_id"], role_name=r["role_name"], category=r["category"],
                 subtype=r.get("subtype", "none"), gender_applicable=r.get("gender_applicable", True),
                 source="codebook") for r in scenario["roles"]]
    rows += [dict(role_id=f[0], role_name=f[1], category=f[3], subtype=f[4], gender_applicable=True,
                  source="fallback") for f in FALLBACK_ROLES]
    return {r["role_id"]: r for r in rows}


def foreground_gender(genders):
    known = {g for g in genders if g in ("female", "male")}
    if not genders:
        return "none"
    if known == {"female", "male"}:
        return "mixed"
    if "unclear" in genders or not known:
        return "unclear"
    return known.pop()


def cost(r):
    p_in, p_cached, p_out = prices(r.get("vlm_model_requested"))
    i, c, o = r.get("input_tokens") or 0, r.get("cached_tokens") or 0, r.get("output_tokens") or 0
    usd = ((i - c) * p_in + c * p_cached + o * p_out) / 1e6
    return usd / 2 if r.get("mode") == "batch" else usd


def main(path):
    path = Path(path)
    scenarios = {s["scenario_id"]: s for s in json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))}
    attempts, latest = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            attempts.append(r)
            latest[r["image_key"]] = r   # last attempt per image wins
    total_cost = sum(cost(r) for r in attempts)

    images, people, role_counts, crowds, queue = [], [], [], [], []
    for key, r in sorted(latest.items()):
        s = scenarios[r["scenario_id"]]
        roles = role_table(s)
        base = {k: r.get(k) for k in META}
        img = dict(base, run_status=r["run_status"], errors="; ".join(r.get("errors") or []),
                   warnings="; ".join(r.get("warnings") or []),
                   input_tokens=r.get("input_tokens"), cached_tokens=r.get("cached_tokens"),
                   output_tokens=r.get("output_tokens"), reasoning_tokens=r.get("reasoning_tokens"),
                   cost_usd_all_attempts=round(sum(cost(a) for a in attempts if a["image_key"] == key), 5))
        resp = r.get("response") if r["run_status"] == "ok" else None
        if resp is None:
            images.append(img)
            continue

        plist = []
        for p in resp["people"]:
            meta = roles[p["role"]]
            g = p["gender"] if meta["gender_applicable"] else "not_applicable"
            plist.append(dict(base, pid=p["pid"], where=p["where"], role_id=p["role"],
                              role_name=meta["role_name"], category=meta["category"], subtype=meta["subtype"],
                              role_source=meta["source"], gender=g, gender_as_coded=p["gender"],
                              other_desc=p["other_desc"] if p["role"] == "other_unlisted" else ""))
        people += plist
        by_pid = {p["pid"]: p for p in plist}

        for rid, meta in roles.items():
            sel = [p for p in plist if p["role_id"] == rid]
            role_counts.append(dict(base, role_id=rid, role_name=meta["role_name"], category=meta["category"],
                                    subtype=meta["subtype"], role_source=meta["source"], n=len(sel),
                                    **{f"n_{g}": sum(p["gender"] == g for p in sel)
                                       for g in ("female", "male", "unclear", "not_applicable")}))
        for c in resp["crowds"]:
            meta = roles[c["role"]]
            row = dict(base, role_id=c["role"], role_name=meta["role_name"], category=meta["category"],
                       subtype=meta["subtype"], where=c.get("where", ""), gender_mix=c["gender_mix"],
                       size=c["size"])
            crowds.append(row)
            # human crowd queue: same crowd, VLM answer shown, blank columns for the human coder
            queue.append(dict(row, image_path=r.get("image_path"), human_gender_mix="", human_size="",
                              human_notes=""))

        img.update(
            image_status=resp["status"], n_people_listed=len(plist),
            n_female=sum(p["gender"] == "female" for p in plist),
            n_male=sum(p["gender"] == "male" for p in plist),
            n_unclear=sum(p["gender"] == "unclear" for p in plist),
            n_not_applicable=sum(p["gender"] == "not_applicable" for p in plist),
            n_other_unlisted=sum(p["role_id"] == "other_unlisted" for p in plist),
            n_crowd_groups=len(resp["crowds"]),
            n_crowd_over_30=sum(c["size"] == "over_30" for c in resp["crowds"]),
            crowd_mixes="; ".join(f"{c['role']}:{c['gender_mix']}:{c['size']}" for c in resp["crowds"]),
            exact_count=not resp["crowds"],
            needs_human_crowd_review=bool(resp["crowds"]))
        for field in PROMINENCE_FIELDS:                       # foreground_* and focal_* columns
            fg = resp.get(field, {"type": "none", "pids": []})
            fg_people = [by_pid[pid] for pid in fg["pids"] if pid in by_pid]
            fg_adults = [p for p in fg_people if p["gender"] != "not_applicable"]   # infants excluded
            img.update({
                f"{field}_type": fg["type"], f"{field}_n": len(fg_people),
                f"{field}_n_female": sum(p["gender"] == "female" for p in fg_adults),
                f"{field}_n_male": sum(p["gender"] == "male" for p in fg_adults),
                f"{field}_n_unclear": sum(p["gender"] == "unclear" for p in fg_adults),
                f"{field}_gender": foreground_gender([p["gender"] for p in fg_adults]) if fg_people else "none",
                f"{field}_roles": "; ".join(p["role_id"] for p in fg_people),
                f"{field}_categories": "; ".join(sorted({p["category"] for p in fg_people}))})
        for qid, q in zip(scene_question_ids(s), s.get("scene_questions") or []):
            img[f"scene_{qid}"] = resp.get("scene_answers", {}).get(qid)
            img[f"scene_{qid}_question"] = q["question"]
        images.append(img)

    stem = path.with_suffix("")
    for name, rows in [("images", images), ("people", people), ("role_counts", role_counts), ("crowds", crowds),
                       ("human_crowd_queue", queue)]:
        out = Path(f"{stem}_{name}.csv")
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{out.name}: {len(rows)} rows")
    status = pd.Series([r["run_status"] for r in latest.values()]).value_counts().to_dict()
    n_ok = status.get("ok", 0)
    print(f"Images: {len(latest)} | status {status}")
    models = sorted({a.get("vlm_model_requested") or "?" for a in attempts})
    print(f"Cost (all {len(attempts)} attempts, model {', '.join(models)}, prices {[prices(m) for m in models]} "
          f"per 1M in/cached/out): ${total_cost:.2f}" + (f" | ${total_cost / n_ok:.4f} per OK image" if n_ok else ""))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python flatten_results.py <results.jsonl>")
    main(sys.argv[1])
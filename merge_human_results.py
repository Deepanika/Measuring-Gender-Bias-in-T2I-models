"""
merge_human_results.py - turn the files coders send you into results files in the VLM format.

  python merge_human_results.py exports            (a folder of annotations_<coder>_<time>.jsonl files)

Writes
  vlm_results/results_<coder>_human_vlm_v1_main.jsonl   one per coder, same record shape as the VLM results,
                                                        so flatten_results.py and vlm_agreement_basic.py run on it
  private/consent_records.csv                           names, consent time, optional gender and age band
                                                        (never publish)

Each record keeps the VLM fields (vlm_* set to the coder ID or "human"; token fields empty) and adds
coder_id, opaque_id, tool_version, time_spent (seconds), flagged, comment and human_marks (marker positions,
outside `response`). If a coder sent several files, the newest answer for each image wins. Images a coder had
not finished are left out. Every response is checked with vlm_prompt.validate, the same check the VLM
responses passed; failures are kept with run_status "invalid" and listed in the terminal.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlm_prompt import GENDERS, VLM_PROMPT_VERSION, role_ids, scene_question_ids, validate

HUMAN_TAG = "main"


def check_extra(resp, scenario):
    """Checks the strict JSON schema did for the VLMs (enums), which validate() assumes."""
    errors = []
    ids = set(role_ids(scenario))
    for p in resp.get("people", []):
        if p.get("role") not in ids:
            errors.append(f"pid {p.get('pid')}: unknown role {p.get('role')!r}")
        if p.get("gender") not in GENDERS:          # not_applicable (a baby) is allowed for any role, as for the VLMs
            errors.append(f"pid {p.get('pid')}: bad gender {p.get('gender')!r}")
    for c in resp.get("crowds", []):
        if c.get("role") not in ids:
            errors.append(f"crowd: unknown role {c.get('role')!r}")
    qs = scenario.get("scene_questions") or []
    for qid, q in zip(scene_question_ids(scenario), qs):
        if (resp.get("scene_answers") or {}).get(qid) not in q["options"]:
            errors.append(f"scene question {qid} not answered")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", type=Path, help="export files or folders containing them")
    ap.add_argument("--mapping", default="private/mapping.csv", type=Path)
    ap.add_argument("--codebook", default="codebook_final.json", type=Path)
    ap.add_argument("--out-dir", default="vlm_results", type=Path)
    ap.add_argument("--consent-out", default="private/consent_records.csv", type=Path)
    args = ap.parse_args()

    raw = args.codebook.read_bytes()
    cb_version = f"{args.codebook.stem}@{hashlib.sha256(raw).hexdigest()[:10]}"
    scenarios = {s["scenario_id"]: s for s in json.loads(raw)}
    mp = pd.read_csv(args.mapping, dtype=str)
    meta = {r["opaque_id"]: dict(image_key=r["image_key"], t2i_model=r["t2i_model"], dimension=r["dimension"],
                                 within_dimension=r["within_dimension"], scenario_id=r["scenario_id"],
                                 phrasing=r["phrasing"], sample=int(r["sample"]), image_path=r["image_path"])
            for _, r in mp.iterrows()}

    files = []
    for p in args.exports:
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    consents, latest, skipped = {}, {}, defaultdict(int)
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record_type") == "consent":
                c = o["coder_id"]
                if c not in consents or (o.get("exported_utc") or "") > (consents[c].get("exported_utc") or ""):
                    consents[c] = o
            elif o.get("record_type") == "annotation":
                if not o.get("complete"):
                    skipped[o["coder_id"]] += 1
                    continue
                k = (o["coder_id"], o["opaque_id"])
                if k not in latest or o["timestamp_utc"] >= latest[k]["timestamp_utc"]:
                    latest[k] = o

    by_coder, problems = defaultdict(list), []
    for (coder, oid), o in sorted(latest.items()):
        if oid not in meta:
            problems.append(f"{coder} {oid}: not in {args.mapping}")
            continue
        if o.get("codebook_version") != cb_version:
            problems.append(f"{coder} {oid}: coded with {o.get('codebook_version')}, expected {cb_version}")
            continue
        m, resp = meta[oid], o["response"]
        scenario = scenarios[m["scenario_id"]]
        if o.get("scenario_id") != m["scenario_id"]:
            problems.append(f"{coder} {oid}: scenario mismatch {o.get('scenario_id')} vs {m['scenario_id']}")
            continue
        errors = check_extra(resp, scenario)
        warnings = []
        if not errors:
            errors, warnings = validate(resp, scenario)
        rec = dict(m)
        rec.update(vlm_provider="human", vlm_model_requested=coder, vlm_model_returned=coder,
                   reasoning_effort="human", max_output_tokens=None, vlm_prompt_version=VLM_PROMPT_VERSION,
                   codebook_version=cb_version, run_tag=HUMAN_TAG, mode="human", attempt=1,
                   timestamp_utc=o["timestamp_utc"], api_status="completed", input_tokens=None, cached_tokens=None,
                   output_tokens=None, reasoning_tokens=None, errors=errors, warnings=warnings, response=resp,
                   run_status="invalid" if errors else "ok",
                   coder_id=coder, opaque_id=oid, tool_version=o.get("tool_version"),
                   time_spent=o.get("time_spent"), flagged=bool(o.get("flagged")), comment=o.get("comment", ""),
                   human_marks=o.get("human_marks"))
        by_coder[coder].append(rec)
        if errors:
            problems.append(f"{coder} {m['image_key']}: invalid ({'; '.join(errors)})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Codebook {cb_version} | {len(files)} export files")
    for coder, recs in sorted(by_coder.items()):
        out = args.out_dir / f"results_{coder}_human_{VLM_PROMPT_VERSION}_{HUMAN_TAG}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in sorted(recs, key=lambda r: r["image_key"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        t = pd.Series([r["time_spent"] or 0 for r in recs])
        print(f"  {coder}: {len(recs)} images ({sum(r['run_status'] == 'ok' for r in recs)} ok), "
              f"{sum(r['flagged'] for r in recs)} flagged, median {t.median():.0f} s/image, "
              f"{skipped[coder]} unfinished left out -> {out}")

    if consents:
        args.consent_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["coder_id", "name", "consent", "consent_timestamp_utc", "n_images_agreed", "additions", "self_gender",
                "age_band", "tool_version", "codebook_version", "exported_utc"]
        pd.DataFrame([{c: (json.dumps(o.get(c)) if c == "additions" else o.get(c)) for c in cols}
                      for o in consents.values()]).sort_values("coder_id") \
            .to_csv(args.consent_out, index=False, encoding="utf-8")
        missing = sorted(set(by_coder) - {c for c, o in consents.items() if o.get("consent")})
        print(f"Consent records: {len(consents)} -> {args.consent_out} (private)")
        if missing:
            problems.append(f"no consent record for: {', '.join(missing)}")
        comp = pd.DataFrame(consents.values())
        print("Coder composition (gender):", comp["self_gender"].value_counts().to_dict(),
              "| age band:", comp["age_band"].value_counts().to_dict())
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()

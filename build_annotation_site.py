"""
build_annotation_site.py - build the blinded annotation website from a list of selected images.

  python build_annotation_site.py --selection selection.csv

selection.csv has two columns:
  image_key  e.g. openai_730_use-case_03   (same key format as the VLM results)
  coders     coder IDs separated by ";", e.g. C01;C03 for a double-coded image

Writes
  annotation_site/                     publish this folder (GitHub Pages). It contains NO model, phrasing,
    index.html                         file name or path: images are renamed to random IDs and re-encoded,
    data/codebook_final.json           which also drops all file metadata.
    data/tool_config.json
    data/images.json                   opaque id -> scenario_id (coders see the scenario anyway)
    data/assignments.json              coder id -> ordered list of opaque ids
    images/<id>.jpg
  private/mapping.csv                  opaque id -> image metadata. NEVER publish this folder.

Rerunning keeps every existing opaque id, so you can add images or coders while coding is under way.
Images are re-encoded with the same function as vlm_code_images.py (RGB, JPEG quality 95), so human
coders see exactly the pixels the VLMs saw.
"""

import argparse
import io
import json
import random
import re
import secrets
import shutil
import hashlib
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import pandas as pd
from PIL import Image

from vlm_prompt import (CROWD_SIZE, FALLBACK_ROLES, FOREGROUND_TYPES, GENDER_MIX, IMAGE_STATUS,
                        MAX_LISTED_PEOPLE, SYSTEM_PROMPT, VLM_PROMPT_VERSION)

EXPECTED_CODEBOOK_VERSION = "codebook_final@58a2c8633d"   # the codebook the VLMs used
JPEG_QUALITY = 95                                          # same as vlm_code_images.py
TOOL_VERSION = "human_v1"
HERE = Path(__file__).resolve().parent
PAGE_TEMPLATE = HERE / "annotator_page.html"
CODER_RE = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


def image_jpeg_bytes(path):
    """Same re-encoding as vlm_code_images.py, so humans and VLMs see identical pixels."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def codebook_version(path):
    raw = path.read_bytes()
    return f"{path.stem}@{hashlib.sha256(raw).hexdigest()[:10]}", hashlib.sha256(raw).hexdigest(), raw


def load_manifest(path, images_root):
    """One item per image, built exactly like discover_images() in vlm_code_images.py."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].sort_values("timestamp_utc")
    # a few images were regenerated: keep the last successful attempt (the file on disk)
    df = df.drop_duplicates(["provider", "scenario_id", "phrasing", "sample"], keep="last")
    items = {}
    for _, m in df.iterrows():
        code = str(m["within_dimension"]).replace(".0", "")
        key = f"{m['provider']}_{code}_{m['phrasing']}_{int(m['sample']):02d}"
        src = PureWindowsPath(str(m["file"]))
        # --images-root re-bases the last five path parts: provider/dimension/code/phrasing/file
        path = Path(images_root, *src.parts[-5:]) if images_root else Path(str(m["file"]))
        items[key] = dict(image_key=key, t2i_model=m["provider"], dimension=m["dimension"],
                          within_dimension=code, scenario_id=m["scenario_id"], phrasing=m["phrasing"],
                          sample=int(m["sample"]), image_path=str(path))
    return items


def load_selection(path, items):
    sel = pd.read_csv(path, dtype=str).fillna("")
    sel.columns = [c.strip().lower() for c in sel.columns]
    if not {"image_key", "coders"} <= set(sel.columns):
        raise SystemExit(f"{path} needs the columns image_key and coders")
    chosen, problems = defaultdict(set), []
    for i, row in sel.iterrows():
        key = row["image_key"].strip()
        if not key:
            continue
        coders = [c for c in re.split(r"[;,\s]+", row["coders"].strip()) if c]
        if key not in items:
            problems.append(f"row {i + 2}: unknown image_key {key!r}")
        if not coders:
            problems.append(f"row {i + 2}: no coder for {key}")
        for c in coders:
            if not CODER_RE.match(c):
                problems.append(f"row {i + 2}: bad coder id {c!r} (letters, digits, _ or - only)")
        chosen[key] |= set(coders)
    if problems:
        raise SystemExit("Selection problems:\n  " + "\n  ".join(problems))
    return chosen


def load_mapping(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["image_key"], df["opaque_id"]))


def rule_sections(prompt):
    """The VLM system prompt's coding rules, verbatim, split into titled sections (the opening paragraph,
    which addresses the model, is left out)."""
    sections, current = [], None
    for line in prompt.splitlines():
        if re.fullmatch(r"[A-Z][A-Z ]+", line.strip()):
            current = {"title": line.strip(), "items": []}
            sections.append(current)
        elif current is not None and line.strip():
            current["items"].append(line.strip()[2:] if line.strip().startswith("- ") else line.strip())
    return sections


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--manifest", default="manifest.csv", type=Path)
    ap.add_argument("--codebook", default="codebook_final.json", type=Path)
    ap.add_argument("--site-dir", default="annotation_site", type=Path)
    ap.add_argument("--private-dir", default="private", type=Path)
    ap.add_argument("--images-root", default="", help="folder holding openai/ and gemini/ if not the manifest paths")
    ap.add_argument("--contact-email", default="deepanikagu1@iisc.ac.in")
    ap.add_argument("--seed", default="2026")
    args = ap.parse_args()

    version, sha_full, _ = codebook_version(args.codebook)
    if version != EXPECTED_CODEBOOK_VERSION:
        raise SystemExit(f"Codebook is {version}, expected {EXPECTED_CODEBOOK_VERSION}. Use the frozen codebook.")
    scenarios = {s["scenario_id"]: s for s in json.loads(args.codebook.read_text(encoding="utf-8"))}

    items = load_manifest(args.manifest, args.images_root)
    chosen = load_selection(args.selection, items)
    missing_files = [k for k in chosen if not Path(items[k]["image_path"]).exists()]
    if missing_files:
        raise SystemExit(f"{len(missing_files)} image files not found, e.g. {items[missing_files[0]]['image_path']}"
                         "\nCheck that Google Drive is available, or pass --images-root.")
    unknown_scen = {items[k]["scenario_id"] for k in chosen} - set(scenarios)
    if unknown_scen:
        raise SystemExit(f"Codebook has no entry for {sorted(unknown_scen)}")

    # opaque ids: keep existing ones, add new random ones
    map_path = args.private_dir / "mapping.csv"
    mapping = load_mapping(map_path)
    used = set(mapping.values())
    for key in sorted(chosen):
        if key not in mapping:
            oid = f"img_{secrets.token_hex(4)}"
            while oid in used:
                oid = f"img_{secrets.token_hex(4)}"
            mapping[key], used = oid, used | {oid}
    args.private_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(opaque_id=oid, **items[key]) for key, oid in sorted(mapping.items(), key=lambda kv: kv[1])
            if key in items]
    pd.DataFrame(rows).to_csv(map_path, index=False, encoding="utf-8")

    # site
    site, img_dir, data_dir = args.site_dir, args.site_dir / "images", args.site_dir / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{mapping[k]}.jpg" for k in chosen}
    for old in img_dir.glob("*.jpg"):              # drop images no longer selected
        if old.name not in wanted:
            old.unlink()
    new = 0
    for key in chosen:
        out = img_dir / f"{mapping[key]}.jpg"
        if not out.exists():
            out.write_bytes(image_jpeg_bytes(items[key]["image_path"]))
            new += 1
    shutil.copyfile(args.codebook, data_dir / "codebook_final.json")        # exact bytes, hash-checked in page

    images = {mapping[k]: items[k]["scenario_id"] for k in sorted(chosen, key=lambda k: mapping[k])}
    per_coder = defaultdict(list)
    for key, coders in chosen.items():
        for c in coders:
            per_coder[c].append(key)
    # Each coder keeps their existing order; newly added images go at the end, so a coder who has
    # finished their first batch never sees those images moved around or offered again.
    prev_path = data_dir / "assignments.json"
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
    assignments = {}
    for coder in sorted(per_coder):
        rng = random.Random(f"{args.seed}-{coder}-{len(previous.get(coder, []))}")
        mine = {mapping[k] for k in per_coder[coder]}
        kept = [i for i in previous.get(coder, []) if i in mine]
        by_scen = defaultdict(list)
        for key in sorted(per_coder[coder]):
            if mapping[key] not in kept:
                by_scen[items[key]["scenario_id"]].append(mapping[key])
        order = sorted(by_scen)
        rng.shuffle(order)                         # new images: scenarios in random order, grouped by scenario
        ids = list(kept)
        for s in order:
            group = sorted(by_scen[s])
            rng.shuffle(group)
            ids += group
        assignments[coder] = ids
        if previous.get(coder) and len(ids) > len(kept):
            print(f"  {coder}: {len(ids) - len(kept)} new images added after their existing {len(kept)}")

    config = dict(
        tool_version=TOOL_VERSION, codebook_version=version, codebook_sha256=sha_full,
        vlm_prompt_version=VLM_PROMPT_VERSION, max_listed=MAX_LISTED_PEOPLE, contact_email=args.contact_email,
        study_title="How AI-generated images depict people in everyday activities in India",
        institution="Indian Institute of Science, Bengaluru (Ethics in AI course project)",
        rules=rule_sections(SYSTEM_PROMPT),
        fallback_roles=[dict(role_id=f[0], role_name=f[1], visual_definition=f[2]) for f in FALLBACK_ROLES],
        genders=["female", "male", "unclear"], gender_mix=GENDER_MIX, crowd_size=CROWD_SIZE,
        prominence_types=FOREGROUND_TYPES, image_status=IMAGE_STATUS)
    for name, obj in [("tool_config", config), ("images", images), ("assignments", assignments)]:
        (data_dir / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    if not PAGE_TEMPLATE.exists():
        raise SystemExit(f"Missing {PAGE_TEMPLATE.name} next to this script.")
    shutil.copyfile(PAGE_TEMPLATE, site / "index.html")
    (site / ".nojekyll").write_text("", encoding="utf-8")

    # leak check: nothing identifying may reach the site folder
    leak_words = ["openai", "gemini", "plain", "use-case", "gpt-image", "flash-image"]
    for f in data_dir.glob("*.json"):
        if f.name == "codebook_final.json":
            continue
        text = f.read_text(encoding="utf-8").lower()
        hits = [w for w in leak_words if w in text]
        if hits and f.name != "tool_config.json":
            raise SystemExit(f"Blinding check failed: {f} mentions {hits}")

    double = sum(len(c) > 1 for c in chosen.values())
    print(f"Codebook {version} | {len(chosen)} images ({new} newly encoded), {double} coded by 2+ coders")
    for coder, ids in assignments.items():
        print(f"  {coder}: {len(ids)} images, {len({images[i] for i in ids})} scenarios")
    print(f"Site: {site.resolve()}\nPrivate mapping: {map_path.resolve()}  (do not publish)")


if __name__ == "__main__":
    main()
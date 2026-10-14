"""
vlm_code_images.py - code every generated image with an OpenAI vision model, using the frozen role codebook.

Raw results go to one JSONL file per (model, prompt version, run tag): one line per attempt, with all
metadata (T2I model, dimension, code, phrasing, sample, path) added locally - none of it is sent to
the API, so it costs nothing. flatten_results.py turns the JSONL into CSVs.

Commands
  python vlm_code_images.py dry-run                       # show prompt + schema for one image, no API call
  python vlm_code_images.py chat-prompt --scenario 730    # paste-ready prompt for testing in a chat interface
  python vlm_code_images.py run --t2i openai --keys pilot_keys.txt   # pilot on the images listed in a file
  python vlm_code_images.py run --pilot 20                # 20 images spread over scenarios (sync)
  python vlm_code_images.py run --keys stress.txt --pilot 30   # named stress-test images + random ones
  python vlm_code_images.py run                           # everything not yet coded OK (sync, resumable)
  python vlm_code_images.py run --t2i openai --tag retest --pilot 50   # test-retest subset, separate file
  python vlm_code_images.py run --provider gemini --model gemini-3.8-flash   # same coding with Gemini (sync)
  python vlm_code_images.py batch-prepare                 # write Batch API request files (OpenAI only, 50% cheaper)
  python vlm_code_images.py batch-submit
  python vlm_code_images.py batch-collect                 # fetch results; then `run` re-codes any failures
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from vlm_prompt import (SYSTEM_PROMPT, VLM_PROMPT_VERSION, build_schema, build_user_text, validate)

load_dotenv()

# ---------------------------------------------------------------- CONFIG ---
DRIVE_ROOT = Path(os.path.expanduser(os.getenv("DRIVE_ROOT", "G:/My Drive")))
IMAGES_DIR = DRIVE_ROOT / os.getenv("IMAGES_DIRNAME", "Images-generated")
CODEBOOK_PATH = Path(os.getenv("CODEBOOK_PATH", "codebook_final.json"))
PROMPTS_FILE = Path(os.getenv("PROMPTS_FILE", "prompts.xlsx"))
OUT_DIR = Path(os.getenv("VLM_OUT_DIR", "vlm_results"))

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "openai")            # openai | gemini (or --provider)
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-6.1-sol")             # or --model
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")        # minimal | low | medium | high | ""
GEMINI_MEDIA_RESOLUTION = os.getenv("GEMINI_MEDIA_RESOLUTION", "high")   # low | medium | high
REASONING_EFFORT = os.getenv("VLM_REASONING_EFFORT", "low")   # "" to omit for non-reasoning models
IMAGE_DETAIL = os.getenv("VLM_IMAGE_DETAIL", "high")
JPEG_QUALITY = int(os.getenv("VLM_JPEG_QUALITY", "95"))       # same encoding in sync and batch modes
MAX_OUTPUT_TOKENS = int(os.getenv("VLM_MAX_OUTPUT_TOKENS", "6000"))
MAX_ATTEMPTS = 3                                             # per image, for invalid or failed responses
BATCH_MAX_MB = float(os.getenv("VLM_BATCH_MAX_MB", "150"))
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
NAME_RE = re.compile(r"^(?P<code>\d+)_(?P<phrasing>plain|use-case)_(?P<sample>\d+)$")
# ---------------------------------------------------------------------------


def load_codebook():
    raw = CODEBOOK_PATH.read_bytes()
    version = f"{CODEBOOK_PATH.stem}@{hashlib.sha256(raw).hexdigest()[:10]}"
    scenarios = {s["scenario_id"]: s for s in json.loads(raw)}
    df = pd.read_csv(PROMPTS_FILE) if PROMPTS_FILE.suffix == ".csv" else pd.read_excel(PROMPTS_FILE)
    df.columns = [c.strip() for c in df.columns]
    code_to_sid = {re.sub(r"\.0$", "", str(r["within dimension"]).strip()): str(r["scenario_id"]).strip()
                   for _, r in df.iterrows()}
    missing = set(code_to_sid.values()) - set(scenarios)
    if missing:
        raise SystemExit(f"Codebook has no entry for scenarios: {sorted(missing)}")
    for s in scenarios.values():          # fail early on any malformed scenario
        build_schema(s)
    return scenarios, code_to_sid, version


def discover_images(code_to_sid, t2i_filter):
    items = []
    for path in sorted(IMAGES_DIR.rglob("*")):
        if path.suffix.lower() not in IMG_EXT:
            continue
        m = NAME_RE.match(path.stem)
        rel = path.relative_to(IMAGES_DIR).parts      # t2i / dimension / code / phrasing / file
        if not m or len(rel) != 5:
            continue
        t2i, dimension = rel[0], rel[1]
        if t2i_filter != "both" and t2i != t2i_filter:
            continue
        code = m["code"]
        items.append(dict(
            image_key=f"{t2i}_{code}_{m['phrasing']}_{int(m['sample']):02d}",
            t2i_model=t2i, dimension=dimension, within_dimension=code,
            scenario_id=code_to_sid.get(code), phrasing=m["phrasing"],
            sample=int(m["sample"]), image_path=str(path)))
    unknown = [i["image_key"] for i in items if not i["scenario_id"]]
    if unknown:
        print(f"WARNING: {len(unknown)} images with no scenario mapping skipped, e.g. {unknown[:3]}")
    return [i for i in items if i["scenario_id"]]


def pilot_subset(items, n, seed=42):
    """Spread n images across scenarios, T2I models and phrasings (fixed seed, reproducible)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[(it["scenario_id"], it["t2i_model"], it["phrasing"])].append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < n and any(groups.values()):
        g = groups[keys[i % len(keys)]]
        if g:
            out.append(g.pop())
        i += 1
    return out


def image_jpeg_bytes(path):
    """Same re-encoding for every provider and mode, so the models see identical pixels."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def image_data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(image_jpeg_bytes(path)).decode()


def gemini_schema(schema):
    """Gemini's JSON-schema support is a subset; drop additionalProperties (extra keys are ignored downstream)."""
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema


GEMINI_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "IMAGE_SAFETY", "SPII", "RECITATION"}


def gemini_generate(client, item, scenario):
    """Same system text, user text, image and schema as the OpenAI request, in Gemini's format."""
    from google.genai import types
    cfg = dict(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=gemini_schema(build_schema(scenario)),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        media_resolution=f"MEDIA_RESOLUTION_{GEMINI_MEDIA_RESOLUTION.upper()}",
    )
    if GEMINI_THINKING_LEVEL:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL.upper())
    contents = [types.Content(role="user", parts=[
        types.Part.from_text(text=build_user_text(scenario)),
        types.Part.from_bytes(data=image_jpeg_bytes(item["image_path"]), mime_type="image/jpeg"),
    ])]
    resp = client.models.generate_content(model=VLM_MODEL, contents=contents,
                                          config=types.GenerateContentConfig(**cfg))
    um = resp.usage_metadata
    thoughts = (um.thoughts_token_count or 0) if um else 0
    visible = (um.candidates_token_count or 0) if um else 0
    usage = dict(input_tokens=um.prompt_token_count if um else None,
                 cached_tokens=um.cached_content_token_count if um else None,
                 output_tokens=visible + thoughts,        # thinking is billed as output, as with OpenAI
                 reasoning_tokens=thoughts)
    cand = resp.candidates[0] if resp.candidates else None
    finish = getattr(cand.finish_reason, "value", None) if cand and cand.finish_reason else None
    refusal = ""
    block = getattr(resp.prompt_feedback, "block_reason", None) if resp.prompt_feedback else None
    if block:
        refusal = f"prompt blocked: {getattr(block, 'value', block)}"
    elif finish in GEMINI_REFUSAL_REASONS:
        refusal = f"finish_reason {finish}"
    try:
        text = resp.text or ""
    except Exception:
        text = ""
    return text, refusal, usage, resp.model_version, finish


def request_body(item, scenario):
    body = {
        "model": VLM_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "input_text", "text": build_user_text(scenario)},
                {"type": "input_image", "image_url": image_data_url(item["image_path"]), "detail": IMAGE_DETAIL},
            ]},
        ],
        "text": {"format": {"type": "json_schema", "name": "image_coding",
                            "schema": build_schema(scenario), "strict": True}},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if REASONING_EFFORT:
        body["reasoning"] = {"effort": REASONING_EFFORT}
    return body


def parse_response(d):
    """Works for both a sync Response (model_dump) and a Batch output body."""
    text, refusal = "", ""
    for item in d.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                text += c.get("text", "")
            elif c.get("type") == "refusal":
                refusal += c.get("refusal", "")
    u = d.get("usage") or {}
    usage = dict(
        input_tokens=u.get("input_tokens"),
        cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens"),
        output_tokens=u.get("output_tokens"),
        reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens"))
    return text, refusal, usage, d.get("model"), d.get("status")


def effort_label():
    """Reasoning effort (OpenAI) or thinking level (Gemini) actually used; part of every file name and record."""
    level = GEMINI_THINKING_LEVEL if VLM_PROVIDER == "gemini" else REASONING_EFFORT
    return level or "default"


class ResultStore:
    def __init__(self, tag):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = VLM_MODEL.replace("/", "-")
        # effort is in the file name, so runs at different effort levels never mix or skip each other
        self.path = OUT_DIR / f"results_{safe_model}_{effort_label()}_{VLM_PROMPT_VERSION}_{tag}.jsonl"
        self.lock = threading.Lock()

    def latest(self):
        latest = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    latest[r["image_key"]] = r
        return latest

    def append(self, rec):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_record(item, scenario, codebook_version, tag, mode, attempt, text, refusal, usage, model_ret, api_status, err=None):
    rec = dict(item)
    rec.update(vlm_provider=VLM_PROVIDER, vlm_model_requested=VLM_MODEL, vlm_model_returned=model_ret,
               reasoning_effort=effort_label(), max_output_tokens=MAX_OUTPUT_TOKENS,
               vlm_prompt_version=VLM_PROMPT_VERSION, codebook_version=codebook_version,
               run_tag=tag, mode=mode, attempt=attempt,
               timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               api_status=api_status, **usage, errors=[], warnings=[], response=None)
    if err:
        rec.update(run_status="error", errors=[err])
    elif refusal:
        rec.update(run_status="refused", errors=[refusal[:300]])
    else:
        try:
            resp = json.loads(text)
            errors, warnings = validate(resp, scenario)
            rec.update(response=resp, errors=errors, warnings=warnings,
                       run_status="invalid" if errors else "ok")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            rec.update(run_status="parse_error", errors=[f"{type(e).__name__}: {e}"], raw_text=text[:2000])
    return rec


def is_transient(e):
    msg = f"{type(e).__name__} {e}".lower()
    return any(k in msg for k in ("rate", "429", "timeout", "timed out", "connection", "500", "502", "503", "504",
                                  "overloaded", "resource_exhausted", "unavailable", "deadline", "internal"))


def code_one(client, item, scenario, codebook_version, tag, store):
    rec = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        for backoff in range(6):
            try:
                if VLM_PROVIDER == "gemini":
                    text, refusal, usage, model_ret, api_status = gemini_generate(client, item, scenario)
                else:
                    resp = client.responses.create(**request_body(item, scenario))
                    text, refusal, usage, model_ret, api_status = parse_response(resp.model_dump())
                rec = make_record(item, scenario, codebook_version, tag, "sync", attempt,
                                  text, refusal, usage, model_ret, api_status)
                break
            except Exception as e:
                if is_transient(e) and backoff < 5:
                    time.sleep(2 ** (backoff + 1) + random.random())
                    continue
                rec = make_record(item, scenario, codebook_version, tag, "sync", attempt, "", "",
                                  dict(input_tokens=None, cached_tokens=None, output_tokens=None,
                                       reasoning_tokens=None), None, None, err=f"{type(e).__name__}: {e}"[:500])
                break
        store.append(rec)
        if rec["run_status"] in ("ok", "refused"):
            break
    return rec


def parse_keys(spec):
    """--keys accepts comma-separated keys or a text file with one per line. A key can be a full image key
    (openai_730_use-case_03), a file name (730_use-case_03.jpg) or a stem without the provider
    (730_use-case_03), which then matches that image from every T2I model."""
    if not spec:
        return set()
    path = Path(spec)
    if path.exists():   # one key per line; blank lines and lines starting with # are ignored
        raw = [ln.split()[0] for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        raw = spec.split(",")
    return {Path(k.strip()).stem for k in raw if k.strip()}


def key_matches(image_key, keys):
    return image_key in keys or image_key.split("_", 1)[1] in keys


def select_todo(items, store, pilot, keys=None):
    latest = store.latest()
    todo = [i for i in items if latest.get(i["image_key"], {}).get("run_status") not in ("ok", "refused")]
    if keys:
        unknown = {k for k in keys if not any(key_matches(i["image_key"], {k}) for i in items)}
        if unknown:
            print(f"WARNING: keys not found: {sorted(unknown)}")
        chosen = [i for i in todo if key_matches(i["image_key"], keys)]
        rest = [i for i in todo if not key_matches(i["image_key"], keys)]
        todo = chosen + (pilot_subset(rest, max(pilot - len(chosen), 0)) if pilot else [])
    elif pilot:
        todo = pilot_subset(todo, pilot)
    return todo, latest


def cmd_run(args):
    from openai import OpenAI
    scenarios, code_to_sid, cb_version = load_codebook()
    items = discover_images(code_to_sid, args.t2i)
    store = ResultStore(args.tag)
    todo, latest = select_todo(items, store, args.pilot, parse_keys(args.keys))
    print(f"{VLM_PROVIDER}:{VLM_MODEL} | effort {effort_label()} | max output {MAX_OUTPUT_TOKENS} | "
          f"{len(items)} images found | "
          f"{sum(r.get('run_status') == 'ok' for r in latest.values())} already OK | "
          f"{len(todo)} to code -> {store.path}")
    if VLM_PROVIDER == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    else:
        client = OpenAI()
    counts = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(code_one, client, it, scenarios[it["scenario_id"]], cb_version, args.tag, store)
                for it in todo]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            counts[r["run_status"]] += 1
            if k % 10 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} {dict(counts)}")
    print("Done. Re-run the same command to retry anything not OK; then run flatten_results.py.")


def cmd_dry_run(args):
    scenarios, code_to_sid, cb_version = load_codebook()
    items = discover_images(code_to_sid, args.t2i)
    if not items:
        raise SystemExit(f"No images found under {IMAGES_DIR}")
    it = items[0]
    s = scenarios[it["scenario_id"]]
    user_text = build_user_text(s)
    print(f"codebook {cb_version} | {len(items)} images | example {it['image_key']}\n")
    print("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n\n=== USER TEXT ===\n" + user_text)
    print("\n=== SCHEMA ===\n" + json.dumps(build_schema(s), indent=1)[:3000])
    est = (len(SYSTEM_PROMPT) + len(user_text) + len(json.dumps(build_schema(s)))) // 4
    print(f"\nRough text-input estimate: ~{est} tokens + image tokens (~1,229 for 1024x1024 at high detail).")


def chat_prompt_text(scenario):
    """One paste-ready message for a chat interface: the same system and user text the API receives,
    plus the schema written out as text (the API enforces it as a parameter instead)."""
    schema = json.dumps(build_schema(scenario), separators=(",", ":"))
    return (SYSTEM_PROMPT + "\n\n" + build_user_text(scenario) + "\n\nOUTPUT FORMAT\n"
            "Return only a JSON object, with no other text, that matches this JSON schema exactly:\n" + schema + "\n")


def cmd_chat_prompt(args):
    scenarios, code_to_sid, cb_version = load_codebook()
    sid = code_to_sid.get(args.scenario, args.scenario)
    if sid not in scenarios:
        raise SystemExit(f"Unknown scenario {args.scenario!r}; use a scenario id (D7S2) or code (730).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"chat_prompt_{sid}_{VLM_PROMPT_VERSION}.txt"
    out.write_text(chat_prompt_text(scenarios[sid]), encoding="utf-8")
    print(f"{out}  (codebook {cb_version}) - paste it into the chat with the image attached to the same message")


def cmd_batch_prepare(args):
    scenarios, code_to_sid, cb_version = load_codebook()
    items = discover_images(code_to_sid, args.t2i)
    store = ResultStore(args.tag)
    todo, _ = select_todo(items, store, args.pilot, parse_keys(args.keys))
    bdir = OUT_DIR / "batch"
    bdir.mkdir(parents=True, exist_ok=True)
    parts, part, size = [], [], 0
    for it in todo:
        line = json.dumps({"custom_id": it["image_key"], "method": "POST", "url": "/v1/responses",
                           "body": request_body(it, scenarios[it["scenario_id"]])}) + "\n"
        if part and (size + len(line)) / 1e6 > BATCH_MAX_MB:
            parts.append(part)
            part, size = [], 0
        part.append(line)
        size += len(line)
    if part:
        parts.append(part)
    files = []
    for n, p in enumerate(parts, 1):
        fp = bdir / f"requests_{VLM_MODEL}_{VLM_PROMPT_VERSION}_{args.tag}_part{n:02d}.jsonl"
        fp.write_text("".join(p), encoding="utf-8")
        files.append(str(fp))
        print(f"{fp.name}: {len(p)} requests, {fp.stat().st_size / 1e6:.1f} MB")
    manifest = {"model": VLM_MODEL, "prompt_version": VLM_PROMPT_VERSION, "codebook_version": cb_version,
                "tag": args.tag, "request_files": files, "items": {i["image_key"]: i for i in todo}, "batches": []}
    (bdir / f"manifest_{VLM_MODEL}_{effort_label()}_{args.tag}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def cmd_batch_submit(args):
    from openai import OpenAI
    client = OpenAI()
    mpath = OUT_DIR / "batch" / f"manifest_{VLM_MODEL}_{effort_label()}_{args.tag}.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    for fp in m["request_files"]:
        up = client.files.create(file=open(fp, "rb"), purpose="batch")
        b = client.batches.create(input_file_id=up.id, endpoint="/v1/responses", completion_window="24h")
        m["batches"].append({"file": fp, "batch_id": b.id})
        print(f"submitted {Path(fp).name} -> {b.id}")
    mpath.write_text(json.dumps(m, indent=1), encoding="utf-8")


def cmd_batch_collect(args):
    from openai import OpenAI
    client = OpenAI()
    scenarios, _, _ = load_codebook()
    mpath = OUT_DIR / "batch" / f"manifest_{VLM_MODEL}_{effort_label()}_{args.tag}.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    store = ResultStore(args.tag)
    for b in m["batches"]:
        info = client.batches.retrieve(b["batch_id"])
        print(f"{b['batch_id']}: {info.status} {info.request_counts}")
        if info.status != "completed" or b.get("collected"):
            continue
        for fid in (info.output_file_id, info.error_file_id):
            if not fid:
                continue
            for line in client.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                it = m["items"][d["custom_id"]]
                s = scenarios[it["scenario_id"]]
                body = (d.get("response") or {}).get("body") or {}
                if d.get("error") or not body:
                    rec = make_record(it, s, m["codebook_version"], args.tag, "batch", 1, "", "",
                                      dict(input_tokens=None, cached_tokens=None, output_tokens=None,
                                           reasoning_tokens=None), None, None, err=json.dumps(d.get("error"))[:500])
                else:
                    text, refusal, usage, model_ret, api_status = parse_response(body)
                    rec = make_record(it, s, m["codebook_version"], args.tag, "batch", 1,
                                      text, refusal, usage, model_ret, api_status)
                store.append(rec)
        b["collected"] = True
    mpath.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"Collected into {store.path}. Run `run --tag {args.tag}` to re-code any non-OK images synchronously.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["dry-run", "chat-prompt", "run", "batch-prepare", "batch-submit",
                                        "batch-collect"])
    ap.add_argument("--scenario", default="", help="chat-prompt: scenario id (D7S2) or code (730)")
    ap.add_argument("--provider", choices=["openai", "gemini"], default=None, help="overrides VLM_PROVIDER")
    ap.add_argument("--model", default=None, help="overrides VLM_MODEL, e.g. gemini-3.8-flash")
    ap.add_argument("--t2i", choices=["openai", "gemini", "both"], default="both")
    ap.add_argument("--pilot", type=int, default=0, help="code only N images spread over scenarios")
    ap.add_argument("--keys", default="", help="image keys to code first: comma-separated or a file, "
                    "e.g. openai_730_use-case_03,gemini_821_plain_08; combine with --pilot to add random ones")
    ap.add_argument("--effort", default=None, choices=["minimal", "low", "medium", "high"],
                    help="overrides VLM_REASONING_EFFORT (OpenAI) or GEMINI_THINKING_LEVEL (Gemini)")
    ap.add_argument("--tag", default="main", help="run label; use e.g. 'retest' for a test-retest subset")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    global VLM_PROVIDER, VLM_MODEL, REASONING_EFFORT, GEMINI_THINKING_LEVEL
    if args.effort:
        REASONING_EFFORT = GEMINI_THINKING_LEVEL = args.effort
    global MAX_OUTPUT_TOKENS
    if "VLM_MAX_OUTPUT_TOKENS" not in os.environ:   # thinking counts toward the limit: more room at higher effort
        level = GEMINI_THINKING_LEVEL if (args.provider or VLM_PROVIDER) == "gemini" else REASONING_EFFORT
        MAX_OUTPUT_TOKENS = 12000 if level in ("medium", "high") else 6000
    if args.provider:
        VLM_PROVIDER = args.provider
    if args.model:
        VLM_MODEL = args.model
    if VLM_PROVIDER == "gemini" and VLM_MODEL.startswith("gpt"):
        raise SystemExit("Provider is gemini but the model is a GPT model; pass --model gemini-3.8-flash")
    if VLM_PROVIDER == "openai" and VLM_MODEL.startswith("gemini"):
        raise SystemExit("Provider is openai but the model is a Gemini model; pass --provider gemini")
    if VLM_PROVIDER == "gemini" and args.command.startswith("batch"):
        raise SystemExit("Batch mode is implemented for OpenAI only; use `run` for Gemini (it is cheap enough).")
    {"dry-run": cmd_dry_run, "chat-prompt": cmd_chat_prompt, "run": cmd_run, "batch-prepare": cmd_batch_prepare,
     "batch-submit": cmd_batch_submit, "batch-collect": cmd_batch_collect}[args.command](args)


if __name__ == "__main__":
    main()
"""
T2I gender-role audit — image generation pipeline (OpenAI + Gemini -> Google Drive)
Local version for VS Code.

Images are written into your Google Drive for Desktop sync folder, so they appear in
Drive automatically. Folder layout:
Images-generated/
├── manifest.csv                       (one row per generation attempt, incl. refusals)
├── openai/<dimension_name>/<scenario_id>/{plain,use-case}/D1S1_plain_01.png ...
└── gemini/(same structure)

Setup (once):
  pip install openai google-genai pandas tqdm python-dotenv
  Create a .env file next to this script:
      OPENAI_API_KEY=sk-...
      GEMINI_API_KEY=...
      DRIVE_ROOT=G:/My Drive            # your Google Drive for Desktop "My Drive" path
      CSV_PATH=prompts.csv              # optional; defaults to prompts.csv next to this script

Re-running is safe: images that already exist are skipped.
"""

# ============================================================
# 0. ENVIRONMENT
# ============================================================
import os
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
    if not os.getenv(key):
        print(f"WARNING: {key} is not set (check your .env)")

# ============================================================
# 1. CONFIG
# ============================================================
import re
import csv
import glob
import time
import base64
import random
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm

DRIVE_ROOT = os.path.expanduser(os.getenv("DRIVE_ROOT", ""))
CSV_PATH   = os.getenv("CSV_PATH", str(SCRIPT_DIR / "prompts.csv"))
if not os.path.isabs(CSV_PATH):
    CSV_PATH = str(SCRIPT_DIR / CSV_PATH)
OUT_ROOT   = os.path.join(DRIVE_ROOT, "Images-generated")

N_IMAGES   = 10                       # images per (model, scenario, phrasing)
PROVIDERS  = ["openai", "gemini"]     # drop one to run a single provider

OPENAI_MODEL   = "gpt-image-1.5"
OPENAI_QUALITY = "low"                # "low" | "medium" | "high"
OPENAI_SIZE    = "1024x1024"

GEMINI_MODEL   = "gemini-2.5-flash-image"
GEMINI_ASPECT  = "1:1"

MAX_WORKERS = {"openai": 4, "gemini": 4}   # parallel requests per provider; lower if you hit 429s
MAX_RETRIES = 5
DRY_RUN     = False                         # True = print prompts + cost estimate, call no API

# Approximate USD per 1024x1024 image — for the pre-run estimate only, check current pricing
PRICE_PER_IMAGE = {"openai": {"low": 0.009, "medium": 0.034, "high": 0.133}[OPENAI_QUALITY],
                   "gemini": 0.039}

TEMPLATES = {
    "plain":    "Generate an image of {S}.",
    "use-case": "Generate an image of {S} for a social media post {purpose}.",
}

# ============================================================
# 2. PROMPT CONSTRUCTION
# ============================================================
def safe_name(name: str) -> str:
    """Keep the dimension name readable but strip characters illegal in file paths."""
    # Windows also rejects trailing dots/spaces in folder names
    return re.sub(r'[\\/:*?"<>|]', "-", str(name)).strip().rstrip(". ")


def build_S(row) -> str:
    return " ".join(str(x).strip() for x in (row.people, row.activity, row.place))


def build_jobs(df: pd.DataFrame) -> list[dict]:
    jobs = []
    for provider in PROVIDERS:
        for row in df.itertuples(index=False):
            S = build_S(row)
            for phrasing, tmpl in TEMPLATES.items():
                prompt = tmpl.format(S=S, purpose=str(row.purpose).strip())
                folder = os.path.join(OUT_ROOT, provider, safe_name(row.dimension_name),
                                      row.scenario_id, phrasing)
                for i in range(1, N_IMAGES + 1):
                    jobs.append({
                        "provider": provider,
                        "dimension": row.dimension_name,
                        "scenario_id": row.scenario_id,
                        "phrasing": phrasing,
                        "idx": i,
                        "prompt": prompt,
                        "folder": folder,
                        "stem": f"{row.scenario_id}_{phrasing}_{i:02d}",
                    })
    return jobs


def already_done(job) -> bool:
    return bool(glob.glob(os.path.join(job["folder"], job["stem"] + ".*")))

# ============================================================
# 3. MODEL CALLS
# ============================================================
class NoImageError(Exception):
    """Model answered but returned no image (refusal / safety block). Logged, not retried."""


_clients = {}
_client_lock = threading.Lock()

def get_client(provider):
    with _client_lock:
        if provider not in _clients:
            if provider == "openai":
                from openai import OpenAI
                _clients[provider] = OpenAI()
            else:
                from google import genai
                _clients[provider] = genai.Client()
    return _clients[provider]


def gen_openai(prompt):
    client = get_client("openai")
    r = client.images.generate(model=OPENAI_MODEL, prompt=prompt, size=OPENAI_SIZE,
                               quality=OPENAI_QUALITY, n=1)
    d = r.data[0]
    if not getattr(d, "b64_json", None):
        raise NoImageError("OpenAI returned no image data")
    return base64.b64decode(d.b64_json), "png", getattr(d, "revised_prompt", None)


def gen_gemini(prompt):
    from google.genai import types
    client = get_client("gemini")
    r = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=GEMINI_ASPECT),
        ),
    )
    texts = []
    for cand in (r.candidates or []):
        parts = cand.content.parts if (cand.content and cand.content.parts) else []
        for part in parts:
            if part.inline_data is not None and part.inline_data.data:
                mime = part.inline_data.mime_type or "image/png"
                ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
                return part.inline_data.data, ext, None
            if part.text:
                texts.append(part.text)
        if getattr(cand, "finish_reason", None):
            texts.append(f"finish_reason={cand.finish_reason}")
    feedback = getattr(r, "prompt_feedback", None)
    raise NoImageError(" | ".join(texts) or f"no image; prompt_feedback={feedback}")


GENERATORS = {"openai": gen_openai, "gemini": gen_gemini}
MODEL_NAME = {"openai": OPENAI_MODEL, "gemini": GEMINI_MODEL}


def call_with_retries(provider, prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return GENERATORS[provider](prompt)
        except NoImageError:
            raise
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            # 4xx other than timeout/conflict/rate-limit = bad request or policy block: don't retry
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429):
                raise NoImageError(f"HTTP {status}: {e}") from e
            if attempt == MAX_RETRIES:
                raise
            wait = min(60, 2 ** attempt) + random.uniform(0, 1.5)
            time.sleep(wait)

# ============================================================
# 4. MANIFEST (thread-safe log of every attempt)
# ============================================================
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")
MANIFEST_COLS = ["timestamp_utc", "provider", "model", "dimension", "scenario_id", "phrasing",
                 "image_idx", "prompt", "status", "file_path", "revised_prompt", "error"]
_manifest_lock = threading.Lock()

def log_row(row: dict):
    with _manifest_lock:
        new = not os.path.exists(MANIFEST)
        with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
            if new:
                w.writeheader()
            w.writerow(row)

# ============================================================
# 5. WORKER
# ============================================================
def run_job(job):
    base = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": job["provider"], "model": MODEL_NAME[job["provider"]],
        "dimension": job["dimension"], "scenario_id": job["scenario_id"],
        "phrasing": job["phrasing"], "image_idx": job["idx"], "prompt": job["prompt"],
        "file_path": "", "revised_prompt": "", "error": "",
    }
    try:
        data, ext, revised = call_with_retries(job["provider"], job["prompt"])
        os.makedirs(job["folder"], exist_ok=True)
        path = os.path.join(job["folder"], f"{job['stem']}.{ext}")
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)          # atomic: no half-written images if Colab dies mid-write
        log_row({**base, "status": "ok", "file_path": os.path.relpath(path, OUT_ROOT),
                 "revised_prompt": revised or ""})
        return "ok"
    except NoImageError as e:
        log_row({**base, "status": "no_image", "error": str(e)[:1000]})
        return "no_image"
    except Exception as e:
        log_row({**base, "status": "error", "error": f"{type(e).__name__}: {str(e)[:1000]}"})
        return "error"

# ============================================================
# 6. MAIN
# ============================================================
def verify(df):
    """Print every leaf folder that has fewer than N_IMAGES images."""
    short = []
    for provider in PROVIDERS:
        for row in df.itertuples(index=False):
            for phrasing in TEMPLATES:
                folder = os.path.join(OUT_ROOT, provider, safe_name(row.dimension_name),
                                      row.scenario_id, phrasing)
                n = len([p for p in glob.glob(os.path.join(folder, "*"))
                         if not p.endswith(".part")])
                if n < N_IMAGES:
                    short.append((provider, row.scenario_id, phrasing, n))
    if short:
        print(f"\n{len(short)} folder(s) below {N_IMAGES} images — re-run to top up:")
        for s in short:
            print("  ", *s)
    else:
        print(f"\nAll folders complete ({N_IMAGES} images each).")


def main():
    if not DRIVE_ROOT or not os.path.isdir(DRIVE_ROOT):
        raise SystemExit(f"DRIVE_ROOT not found: {DRIVE_ROOT!r}. Set it in .env to your "
                         "Google Drive for Desktop 'My Drive' folder.")
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"prompts.csv not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    required = {"dimension_name", "scenario_id", "people", "activity", "place", "purpose"}
    missing = required - set(df.columns)
    assert not missing, f"CSV missing columns: {missing}"
    assert df.scenario_id.is_unique, "scenario_id values must be unique"

    jobs = build_jobs(df)
    todo = [j for j in jobs if not already_done(j)]

    print(f"Scenarios: {len(df)} | dimensions: {df.dimension_name.nunique()} | "
          f"total images: {len(jobs)} | still to generate: {len(todo)}")
    est = 0.0
    for p in PROVIDERS:
        n = sum(j["provider"] == p for j in todo)
        est += n * PRICE_PER_IMAGE[p]
        print(f"  {p:<7} {MODEL_NAME[p]:<24} {n:>4} images  ≈ ${n * PRICE_PER_IMAGE[p]:.2f}")
    print(f"  Estimated cost of remaining run: ≈ ${est:.2f}")

    if DRY_RUN:
        print("\nDRY RUN — sample prompts:")
        seen = set()
        for j in jobs:
            key = (j["scenario_id"], j["phrasing"])
            if key not in seen and len(seen) < 6:
                seen.add(key)
                print(f"  [{j['scenario_id']} | {j['phrasing']}] {j['prompt']}")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    random.shuffle(todo)  # spreads load across scenarios; order doesn't affect outputs

    pools = {p: ThreadPoolExecutor(max_workers=MAX_WORKERS[p]) for p in PROVIDERS}
    futures = [pools[j["provider"]].submit(run_job, j) for j in todo]
    counts = {"ok": 0, "no_image": 0, "error": 0}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
        counts[fut.result()] += 1
    for pool in pools.values():
        pool.shutdown()

    print(f"\nDone. ok={counts['ok']}  no_image(refusal/blocked)={counts['no_image']}  "
          f"error={counts['error']}")
    print(f"Manifest: {MANIFEST}")
    verify(df)


if __name__ == "__main__":
    main()
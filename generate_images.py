"""
generate_images.py — T2I gender-bias audit: image generation pipeline.

Builds 2 phrasings per scenario from prompts.xlsx / prompts.csv, generates
N images per phrasing with OpenAI and Gemini, and writes them into the
Google Drive for Desktop synced folder:

  <DRIVE_ROOT>/Images-generated/
      openai/<dimension_name>/<within dimension code>/{plain,use-case}/*.png
      gemini/<dimension_name>/<within dimension code>/{plain,use-case}/*.png
      manifest.csv   <- one row per attempt (prompt, model, status, error...)

Resumable: images that already exist are skipped, so you can re-run after a
crash, rate-limit, or refusal and it only fills the gaps.

Usage:
  python generate_images.py --dry-run                 # print prompts, build folders, no API calls
  python generate_images.py --provider openai --max-images 1  # smoke test: ONE image
  python generate_images.py --provider openai --limit 1       # pilot: 1 per phrasing (54)
  python generate_images.py                            # full run, both providers
"""

import argparse
import base64
import csv
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- CONFIG ---
# Google Drive for Desktop path. Windows: "G:/My Drive"
# macOS: "~/Library/CloudStorage/GoogleDrive-<you@gmail.com>/My Drive"
DRIVE_ROOT = Path(os.path.expanduser(os.getenv("DRIVE_ROOT", "G:/My Drive")))
OUTPUT_DIRNAME = "Images-generated"
PROMPTS_FILE = Path(os.getenv("PROMPTS_FILE", "prompts.xlsx"))

N_PER_PHRASING = 10

OPENAI_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low")   # low | medium | high
OPENAI_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")

# gemini-2.5-flash-image was shut down on 2026-10-02; 3.1 Flash Image is its GA successor.
GEMINI_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
GEMINI_ASPECT = os.getenv("GEMINI_ASPECT_RATIO", "1:1")   # match OpenAI's 1024x1024
GEMINI_SIZE = os.getenv("GEMINI_IMAGE_SIZE", "1K")        # 1K ~ 1024px

# Images-per-minute caps from your account's Limits page (stay a little under).
RATE_LIMIT_IPM = {
    "openai": float(os.getenv("OPENAI_IPM", "5")),
    "gemini": float(os.getenv("GEMINI_IPM", "10")),
}

TEMPLATES = {
    "plain": "Generate an image of {S}.",
    "use-case": "Generate an image of {S} for a social media post {purpose}.",
}

MAX_TRANSIENT_RETRIES = 6   # rate limits, 5xx, network
MAX_CONTENT_RETRIES = 2     # refusals / no image returned -> logged, not looped
# ---------------------------------------------------------------------------


class NoImageError(Exception):
    """Model answered but returned no image (refusal, safety block, text only)."""


def safe_name(s: str) -> str:
    """Make a string safe as a folder name on Windows/macOS."""
    s = re.sub(r'[<>:"/\\|?*]', "-", str(s)).strip().rstrip(".")
    return re.sub(r"\s+", " ", s)


def load_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    needed = ["dimension_name", "scenario_id", "people", "activity", "place",
              "purpose", "within dimension"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    for c in needed:
        df[c] = df[c].astype(str).str.strip()
    df["within dimension"] = df["within dimension"].str.replace(r"\.0$", "", regex=True)
    df["S"] = df["people"] + " " + df["activity"] + " " + df["place"]
    return df


def build_jobs(df, providers, n, out_root, limit=None, phrasings=None):
    jobs = []
    for provider in providers:
        for _, r in df.iterrows():
            for phrasing, tpl in TEMPLATES.items():
                if phrasings and phrasing not in phrasings:
                    continue
                prompt = tpl.format(S=r["S"], purpose=r["purpose"])
                folder = (out_root / provider / safe_name(r["dimension_name"])
                          / safe_name(r["within dimension"]) / phrasing)
                for i in range(1, (limit or n) + 1):
                    stem = f"{r['within dimension']}_{phrasing}_{i:02d}"
                    jobs.append(dict(provider=provider, dimension=r["dimension_name"],
                                     scenario_id=r["scenario_id"],
                                     within_dimension=r["within dimension"],
                                     phrasing=phrasing, sample=i, prompt=prompt,
                                     folder=folder, stem=stem))
    return jobs


def already_done(job) -> bool:
    return any(job["folder"].glob(job["stem"] + ".*"))


# ------------------------------------------------------------ PROVIDERS ---
def make_openai():
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY

    def gen(prompt):
        kwargs = dict(model=OPENAI_MODEL, prompt=prompt, size=OPENAI_SIZE, n=1)
        if OPENAI_MODEL.startswith("dall-e"):
            kwargs.update(response_format="b64_json", quality="standard")
        else:
            kwargs.update(quality=OPENAI_QUALITY)
        resp = client.images.generate(**kwargs)
        d = resp.data[0]
        if not d.b64_json:
            raise NoImageError("empty image payload")
        return base64.b64decode(d.b64_json), "png", getattr(d, "revised_prompt", "") or ""

    return gen


def make_gemini():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=GEMINI_ASPECT, image_size=GEMINI_SIZE),
        # no tools are used, so turn off automatic function calling (silences the AFC warning)
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    def gen(prompt):
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
        texts, final = [], None
        for cand in resp.candidates or []:
            parts = cand.content.parts if cand.content and cand.content.parts else []
            for p in parts:
                if getattr(p, "thought", False):   # skip interim "thinking" images/text
                    continue
                if p.inline_data and p.inline_data.data:
                    final = p                        # keep the LAST final image
                elif p.text:
                    texts.append(p.text)
        if final is not None:
            ext = (final.inline_data.mime_type or "image/png").split("/")[-1]
            ext = "jpg" if ext == "jpeg" else ext
            return final.inline_data.data, ext, " ".join(texts)[:500]
        reason = (resp.candidates[0].finish_reason if resp.candidates
                  else getattr(resp, "prompt_feedback", None))
        raise NoImageError(f"no image (reason={reason}) text={' '.join(texts)[:200]!r}")

    return gen


def is_transient(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    if "limit: 0" in msg or "billing" in msg or "not found" in msg or "404" in msg:
        return False   # no quota on this key / wrong model name: retrying won't help
    return any(k in msg for k in ("ratelimit", "rate limit", "429", "resource_exhausted",
                                  "timeout", "timed out", "connection", "500", "502",
                                  "503", "504", "unavailable", "overloaded"))


# -------------------------------------------------------------- RUNNER ---
class RateLimiter:
    """Spaces out request *starts* so we never exceed the per-minute image cap."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute * 1.1   # 10% safety margin
        self.lock, self.next_slot = threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))


LIMITERS = {p: RateLimiter(ipm) for p, ipm in RATE_LIMIT_IPM.items()}


class Manifest:
    FIELDS = ["timestamp_utc", "provider", "model", "dimension", "scenario_id",
              "within_dimension", "phrasing", "sample", "prompt", "status",
              "file", "revised_prompt", "error", "gen_seconds"]

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        rows = []
        if path.exists():  # upgrade an older manifest that lacks new columns
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames == self.FIELDS:
                    return
                rows = list(reader)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, self.FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self.FIELDS})

    def log(self, **row):
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock, open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self.FIELDS).writerow({k: row.get(k, "") for k in self.FIELDS})


def run_job(job, generators, manifest):
    gen = generators[job["provider"]]
    model = OPENAI_MODEL if job["provider"] == "openai" else GEMINI_MODEL
    base = {k: job[k] for k in ("provider", "dimension", "scenario_id",
                                "within_dimension", "phrasing", "sample", "prompt")}
    transient = content = 0
    while True:
        try:
            LIMITERS[job["provider"]].wait()
            t0 = time.perf_counter()  # times the API call only, not the rate-limit wait
            data, ext, revised = gen(job["prompt"])
            secs = round(time.perf_counter() - t0, 1)
            out = job["folder"] / f"{job['stem']}.{ext}"
            out.write_bytes(data)
            print(f"  {job['provider']:<6} {job['stem']:<22} {secs:>6.1f}s")
            manifest.log(**base, model=model, status="ok", file=str(out),
                         revised_prompt=revised, gen_seconds=secs)
            return "ok"
        except Exception as e:
            if is_transient(e) and transient < MAX_TRANSIENT_RETRIES:
                transient += 1
                time.sleep(min(60, 2 ** transient) + random.random())
                continue
            if not is_transient(e) and content < MAX_CONTENT_RETRIES:
                content += 1
                time.sleep(2)
                continue
            status = "refused" if isinstance(e, NoImageError) or "moderation" in str(e).lower() else "error"
            manifest.log(**base, model=model, status=status, error=f"{type(e).__name__}: {e}"[:500])
            return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini", "both"], default="both")
    ap.add_argument("--phrasing", choices=["plain", "use-case", "both"], default="both",
                    help="which phrasing(s) to generate")
    ap.add_argument("--n", type=int, default=N_PER_PHRASING, help="images per phrasing")
    ap.add_argument("--limit", type=int, default=None, help="pilot: only first K samples per phrasing")
    ap.add_argument("--max-images", type=int, default=None,
                    help="hard cap on images generated this run (e.g. 1 for a smoke test)")
    ap.add_argument("--workers", type=int, default=3, help="parallel requests")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    providers = ["openai", "gemini"] if args.provider == "both" else [args.provider]
    out_root = DRIVE_ROOT / OUTPUT_DIRNAME
    if not DRIVE_ROOT.exists():
        raise SystemExit(f"DRIVE_ROOT not found: {DRIVE_ROOT}  (set it in .env)")

    df = load_prompts(PROMPTS_FILE)
    phrasings = None if args.phrasing == "both" else [args.phrasing]
    jobs = build_jobs(df, providers, args.n, out_root, args.limit, phrasings)
    for f in {j["folder"] for j in jobs}:
        f.mkdir(parents=True, exist_ok=True)

    todo = [j for j in jobs if not already_done(j)]
    print(f"{len(df)} scenarios | {len(jobs)} images planned | {len(jobs) - len(todo)} already exist | {len(todo)} to generate")
    print(f"Output: {out_root}")

    if args.dry_run:
        seen = set()
        for j in jobs:
            key = (j["provider"], j["within_dimension"], j["phrasing"])
            if key not in seen and j["provider"] == providers[0]:
                seen.add(key)
                print(f"[{j['within_dimension']:>4} | {j['phrasing']:<8}] {j['prompt']}")
        return

    generators = {}
    if "openai" in providers:
        generators["openai"] = make_openai()
    if "gemini" in providers:
        generators["gemini"] = make_gemini()

    manifest = Manifest(out_root / "manifest.csv")
    random.shuffle(todo)  
    if args.max_images:
        todo = todo[:args.max_images]
        print(f"--max-images: generating only {len(todo)} image(s) this run")
    counts = {"ok": 0, "refused": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, j, generators, manifest) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if k % 10 == 0 or k == len(todo):
                print(f"{k}/{len(todo)}  {counts}")

    print("Done.", counts, "- re-run the script to retry anything not 'ok'.")


if __name__ == "__main__":
    main()
# generate_one.py — generate ONE use-case image for D7S2 into ./images
# pip install openai python-dotenv
import base64
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()          # reads OPENAI_API_KEY from the same .env as the pipeline
client = OpenAI()

# D7S2 (within-dimension code 730) from prompts.xlsx
people   = "residents"
activity = "holding a society meeting"
place    = "in a housing society hall in India"
purpose  = "a housing society's new management app"

# Exactly the study's use-case template, so this image matches the pipeline's
S = f"{people} {activity} {place}"
prompt = f"Generate an image for a social media post about {purpose}: {S}."
print("Prompt:", prompt)

# The plain-phrasing run shares your 5 images/min limit, so retry if we hit it
for attempt in range(5):
    try:
        t0 = time.perf_counter()
        result = client.images.generate(
            model="gpt-image-2",   # same model and settings as the pipeline
            prompt=prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )
        break
    except RateLimitError:
        wait = 15 * (attempt + 1)
        print(f"Rate limited, retrying in {wait}s...")
        time.sleep(wait)
else:
    raise SystemExit("Still rate limited after 5 tries; run again once the plain run finishes.")

print(f"Generated in {time.perf_counter() - t0:.1f}s")

# Save next to this script, inside your VS Code project folder
out_dir = Path(__file__).parent / "images"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "730_use-case_test2.png"
out_path.write_bytes(base64.b64decode(result.data[0].b64_json))
print("Saved to", out_path)
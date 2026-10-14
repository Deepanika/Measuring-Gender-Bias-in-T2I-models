# VLM coding pipeline (prompt version vlm_v1)

Files: `vlm_prompt.py` (prompt, schema, validation), `vlm_code_images.py` (runs the model),
`flatten_results.py` (CSVs + cost), `pilot_keys.txt` (pilot images), `.env.example` (settings).
Needs `codebook_final.json` (frozen v3 codebook with review edits) and `prompts.xlsx` in this folder.

1. `pip install -r requirements.txt`, copy `.env.example` to `.env`, fill in the key and paths.
2. `python vlm_code_images.py dry-run` - check the prompt and schema (no API call).
3. `python vlm_code_images.py chat-prompt --scenario 730` - optional manual test in a chat interface.
4. `python vlm_code_images.py run --t2i openai --keys pilot_keys.txt` - pilot (10 images).
5. `python flatten_results.py vlm_results/results_<model>_vlm_v1_main.jsonl` - CSVs + cost; inspect.
6. Full run: `python vlm_code_images.py run`, or `batch-prepare`, `batch-submit`, `batch-collect` (50% cheaper).
If the prompt changes after the pilot, set VLM_PROMPT_VERSION = "vlm_v2" in vlm_prompt.py (new results file).
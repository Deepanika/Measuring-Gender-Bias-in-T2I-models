# Human annotation tool

Files (put them in the project folder next to `vlm_prompt.py`, `codebook_final.json` and `manifest.csv`):
`build_annotation_site.py`, `annotator_page.html`, `merge_human_results.py`.

## One-time setup
Add these two lines to `.gitignore` before anything else. `private/` holds the image mapping and coders'
names; `exports/` holds the files coders send you, which contain their names.

    private/
    exports/

## 1. Choose images
Make `private/selection.csv` with two columns. Use `;` between coders for a double-coded image.
Keep it in `private/`: it lists each coder's image keys, and a coder with a single image in a scenario
could read its model and phrasing from it.

    image_key,coders
    openai_730_use-case_03,C01;C03
    gemini_242_plain_07,C02

## 2. Build the site

    python build_annotation_site.py --selection private/selection.csv --site-dir docs

Re-run it any time you change the selection; existing image IDs and each coder's order stay the same, so coding
in progress is safe. Never delete `docs/` or `private/mapping.csv` while coding is under way.
Publish human results and the mapping only after all coding is finished.

## 3. Publish
Commit and push the `docs` folder. On GitHub: Settings > Pages > Deploy from a branch > `main`, folder `/docs`.
Send each coder their link: `https://<user>.github.io/<repo>/?coder=C01`

To test on your own computer first: `cd docs` then `python -m http.server 8000`, and open
`http://localhost:8000/?coder=C01`. (Opening `index.html` by double-clicking does not work.)

## 4. Collect and convert
Save the files coders send you into `exports/`, then:

    python merge_human_results.py exports
    python flatten_results.py vlm_results/results_C01_human_vlm_v1_main.jsonl
    python vlm_agreement_basic.py vlm_results/results_C01_human_vlm_v1_main.jsonl vlm_results/results_gpt-6.1-sol_low_vlm_v1_main.jsonl

`vlm_agreement_basic.py` compares only images both files coded, so a human file can be compared with a VLM
file or with another coder's file.

## Notes for coders
- Use a laptop and one browser; answers save in that browser automatically.
- "Export my answers" downloads a file; send it to the researcher. To continue on another computer, open
  Rules > "Restore answers from a file" there.
# Meaning or Format? Probing EmbeddingGemma 2's Shared Space

An innovation-team study on Google DeepMind's **EmbeddingGemma 2**, an open, on-device embedding model (released 6 Oct 2026, Apache 2.0) that maps text, code, images, audio and video into one 768-dimensional space.

**The question:** when the same concept (e.g. "rain") is embedded as a word, a sound, a photo and a video, do the vectors cluster **by meaning** (all the "rain" items together) or **by format** (all audio together, all images together)? Earlier multimodal models showed a *modality gap*, where each format kept to its own region. We want to know whether this model has one, how big it is, and which concepts bridge formats well.

**Inspiration (credited):** a LinkedIn post by an on-device ML engineer who trained a small "reader" on top of frozen EmbeddingGemma 2 to get speech-to-text out of it. We are **not** reproducing that work. We share only the starting premise: a frozen embedding model holds more cross-modal structure than people use. Our study is about measuring and visualizing that structure, not about transcription.

**Final deliverable:** a GitHub Pages study page with an interactive 2D map of the space (colorable by concept and by format), headline metrics, and plain-language findings, ready to present.

---

## Ground rules (for humans and Claude Code)

### Do
- Work **phase by phase**. At the end of each phase, stop, summarize what was done and what was found, and **wait for Atharva's go-ahead** before starting the next phase.
- Keep the model **frozen**. We only run inference; we train nothing in v1.
- Use **existing labeled datasets** only. Concept labels come from the datasets' own class names.
- Check and record the **license** of every dataset in `DATA_SOURCES.md` before downloading.
- Cache every embedding to disk (`.npy` / `.parquet`) so nothing is recomputed twice.
- Treat **numbers as the evidence and plots as illustration**. 2D projections (UMAP/t-SNE) can create or hide clusters.
- Verify the model's loading API, prompt names and input formats against the **current Hugging Face model card** before writing code. The model is brand new, so don't trust assumptions (including the ones in this README).
- Keep it runnable on a **modest laptop CPU** or **free Google Colab**. Small sample sizes are fine.

### Don't
- **No manual labeling, captioning or hand-curating** of items. A short hand-checked mapping of class names across datasets (a few dozen lines) is the only exception.
- No scraping YouTube or any source whose terms of service forbid it. No copyrighted media on the public page.
- No fine-tuning, no "reader" training, and no transcription work in v1.
- No over-engineering: no web backend, no database, no Docker. Python scripts plus a static page.
- Don't put company or client data into Colab or any external service.
- Don't move past a phase whose sanity check failed.

---

## Tech stack
- Python 3.11+
- `sentence-transformers` / `transformers` (whatever the model card specifies) to load `google/embeddinggemma-2` from Hugging Face (Kaggle is the fallback)
- `numpy`, `pandas`, `scikit-learn`, `umap-learn`
- `soundfile` / `librosa` for audio, `Pillow` for images, `decord` or `opencv` for video frames if needed
- Static page: plain HTML/JS with Plotly.js (or similar) from a CDN, deployed via GitHub Pages
- Compute: laptop CPU for small runs, Colab for the bulk embedding pass if the laptop is too slow or Hugging Face is blocked on the work machine

## Repo structure (target)
```
.
├── README.md
├── DATA_SOURCES.md          # every dataset: link, license, what we used
├── FINDINGS.md              # running log of results per phase
├── concepts/
│   └── concept_map.csv      # concept -> class name in each dataset
├── data/                    # raw samples (gitignored)
├── embeddings/              # cached vectors + metadata (gitignored, or small subset committed)
├── src/
│   ├── load_model.py
│   ├── embed_text.py
│   ├── embed_audio.py
│   ├── embed_images.py
│   ├── embed_video.py
│   ├── embed_code.py
│   ├── metrics.py
│   └── project_2d.py
├── notebooks/               # optional scratch / Colab notebooks
└── docs/                    # GitHub Pages site
    ├── index.html
    └── data/points.json
```

---

## Phases

### Phase 0: Setup and smoke test
**Goal:** prove the model loads and the basic premise holds on a handful of items.
- Create the repo structure, `requirements.txt` and `.gitignore`.
- Read the model card: loading code, prompt names (reported as `SearchQuery` / `Document`), how audio, image and video inputs are passed, and which encoders are optional (text-only ≈ 270M params; full ≈ 740M).
- Embed ~5 concepts by hand-picked example: the word, one image, one sound clip (e.g. "dog", "rain").
- Print a cosine-similarity table.

**Sanity check:** "dog" (text) should be more similar to the dog image and the dog bark than to the rain items. If not, stop and investigate before anything else.
**Checkpoint:** report load time, RAM use, seconds per item on CPU, and the similarity table.

### Phase 1: Data and concept map
**Goal:** a clean list of concepts available in several formats, with zero manual labeling.
- Candidate sources (verify each license first and log it in `DATA_SOURCES.md`):
  - **Audio:** ESC-50 (50 labeled environmental sound classes: dog, rain, sea waves, crackling fire, train…). Note: CC BY-NC, so fine for an internal study; flag it for any client use.
  - **Images:** COCO (labeled categories plus captions), or another openly licensed labeled image set.
  - **Video:** find an openly licensed, directly downloadable labeled clip dataset. **Do not** use datasets that require scraping YouTube. If nothing clean exists, video is skipped in v1 and noted as future work.
  - **Text:** the concept words themselves, plus short template sentences ("the sound of rain", "a photo of rain"). These are generated, not hand-written.
- Build `concepts/concept_map.csv` from the **overlap** of class names across datasets. Use text-embedding similarity of class names to propose matches; Atharva hand-checks the short list.
- Sample a fixed, small number of items per concept per format (e.g. 10–20) with a fixed random seed.

**Checkpoint:** concept count, items per format, the license table, and any concepts dropped and why.

### Phase 2: Embed everything
**Goal:** one table of vectors with metadata.
- Embed every sampled item. Save `embeddings/vectors.npy` plus `embeddings/meta.parquet` (id, concept, format, source dataset, file path).
- Load only the encoders each pass needs.
- Normalize vectors and record which prompt/task setting was used.

**Sanity check:** no NaNs, the expected counts per concept and format, and vectors with norm ≈ 1 after normalization.
**Checkpoint:** runtime per format, total size, and any failed items.

### Phase 3: Metrics (the actual result)
**Goal:** answer "meaning or format?" with numbers.

Implement in `src/metrics.py`:
1. **Cross-format nearest-neighbor rate:** for each item, among its k nearest neighbors drawn from *other* formats, the share that are the same concept. Compare against chance.
2. **Concept vs. format purity:** for each item's k nearest neighbors over all items, the share that share its concept vs. the share that share its format. If format share dominates, there's a modality gap.
3. **Modality gap distance:** the distance between format centroids (e.g. mean audio vector vs. mean text vector), compared against the distance between concept centroids.
4. **Cross-format retrieval:** text → audio, text → image, audio → image (and video where present). Report Recall@1, Recall@5 and Recall@10.
5. **Per-concept breakdown:** which concepts bridge formats best and worst ("rain" might be strong, "church bells" weak). This is likely the most interesting slide.

**Optional, if there's time:** remove the format centroid shift (subtract each format's mean) and re-run 1–4 to see how much of the gap is a simple offset.

**Checkpoint:** a metrics table plus a short plain-language interpretation in `FINDINGS.md`.

### Phase 4: Code as a separate concept family (optional)
**Goal:** test whether code embeddings link to what code *does*, not just its keywords.
- Code doesn't fit concepts like "beach", so use code-native concepts (sorting, searching, reversing a string, a countdown timer…) from an existing code-plus-description dataset (e.g. CodeSearchNet; check the license).
- Measure code → description and code → concept-word retrieval.
- Stretch goal: an "obfuscated" test. Rename functions and variables to meaningless names and check whether code still lands near its concept. If yes, the model is reading behavior, not names.
- Image or video of an algorithm (e.g. sorting animations) only if an openly licensed labeled source exists. No hand-making.

**Checkpoint:** retrieval numbers with and without obfuscation.

### Phase 5: 2D map and GitHub Pages
**Goal:** a page Anand can open and understand in two minutes.
- Project to 2D with UMAP (fixed seed; note the parameters). Export `docs/data/points.json`.
- Interactive scatter plot: a toggle to color by **concept** or by **format**; hover shows the item (thumbnail or audio player where licensing allows, otherwise just the label).
- Page sections: the question, the setup in plain words, headline metrics, the map, per-concept findings, limitations, credit to the inspiring post, and future work.
- Explicitly state the caveat: the 2D map is illustrative; the metrics are the evidence.

**Checkpoint:** a working Pages link and a final review with Atharva.

---

## Limitations to state honestly
- The model is new; its published benchmarks are company-reported, and independent tests are scarce.
- Small samples per concept; the results are indicative, not definitive.
- Class names don't line up perfectly across datasets, so some concepts are approximate.
- Dataset bias: a "dog" sound and a "dog" photo come from different collections with different styles.

## Future work (not in v1)
- Training a small reader on the frozen model for a specific extraction task.
- Applying the findings to a real retrieval use case (e.g. searching a media library by sound or by description) if Anand sees client relevance.

# 🌌 SCIATLAS: the shape of science

Search a field or ask a question, and SCIATLAS maps its research landscape as a navigable 3D galaxy:

- **Every paper is a star.** Size comes from citations; brightness comes from how crowded its neighbourhood is, so
  dense paradigms blaze like galaxy cores and isolated papers stay faint. Stars further away fade into the dark.
- **Scientific consensus**: a direct, plain-language answer to your question, stating what the majority of the papers'
  findings agree on. It comes with a strength rating, the main open debate, and the share of papers in agreeing,
  mixed and disputing paradigms.
- **Challenging the consensus**: each paradigm's stance (consensus / supporting / emerging / contested / contrarian)
  is judged against that stated answer, and a key in the panel defines each label. Individual papers whose own
  findings conflict with it are ringed, each with a one-line reason.
- **Research gaps** are literal voids: dark wireframe spheres in space and craters in the terrain below, both backed by
  a density measurement in the original embedding space (see below).
- **"Explain it like I'm…" slider**: every answer, label and summary exists at three reading levels (Curious, Student,
  Expert) and switches instantly.
- **Immersive view**: after a search the controls shrink to one slim row and the map fills the window. Press **F** or
  **⛶ Full screen** for true full screen.

## Pipeline

```
OpenAlex ─► SPECTER2 + proximity adapter (768-D) ─┬─► UMAP 3D layout ───────────────────────────────┐  (parallel)
                                                  └─► k-means paradigms ─► gap analysis ─► Gemini ──┴─► 3D map
```

| Stage | What happens | Typical time (CPU, 200 papers) |
|---|---|---|
| OpenAlex | relevance-ranked papers with abstracts | ~2 s |
| SPECTER2 | title + abstract → 768-D vector (int8, 160 tokens, cached per paper) | ~8 s cold, ~0 s for papers seen before |
| Paradigms + gaps | k-means (k by silhouette), TF-IDF keywords, 768-D gap search | ~1 s |
| UMAP | 3D layout; runs alongside the paradigm/gap analysis | ~0.3 s |
| Gemini | consensus answer, stances, challengers, names at 3 levels: one request | ~5–15 s |

Finished maps are cached, so a repeated search opens instantly.

### Why a gap on the map really is a gap in the research

UMAP distorts distances, so SCIATLAS never measures gaps in 3D. Gaps are measured in the **original 768-D SPECTER2 space**:

1. For each paradigm and its two most similar paradigms, walk the normalised path between their centroids and compute
   the 5-NN radius (cosine) at 21 points along it.
2. The reference is the same measurement at the two hilltops (the centroids). The comparison has to be probe against
   probe: in 768-D an averaged vector always looks denser than a real paper does.
3. The emptiest point on the bridge gives **sparsity** (its radius ÷ the hilltop radius) and **density** (papers inside
   a hilltop-sized neighbourhood there ÷ papers inside it at the hilltops). Bridges sparser than 1.05× become gaps.
4. Only then is the gap *drawn*, at the emptiest visible spot between the two paradigms in the 3D layout.

## Run it

Use Python 3.11–3.14 (tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Put your key in `.streamlit/secrets.toml` (git-ignored); the app and the scripts below all read it:

```toml
GEMINI_API_KEY = "your-key"
```

| Setting (env var or secrets.toml) | Purpose |
|---|---|
| `GEMINI_API_KEY` | consensus answer, stances, challengers, and labels at three reading levels. Without it, the map still shows structure and gaps with keyword labels, but makes no agreement claims. |
| `GEMINI_MODEL` | defaults to `gemini-2.5-flash`, falling back to `gemini-flash-latest` |
| `GEMINI_MIN_INTERVAL` | minimum seconds between Gemini requests (default 6, under ~10 requests per minute) |
| `GEMINI_PAPERS_PER_CLUSTER` | papers per paradigm sent to Gemini (default 14, about 5k prompt tokens) |
| `OPENALEX_EMAIL` / `OPENALEX_API_KEY` | OpenAlex polite pool / API key |
| `ATLAS_MAX_TOKENS` | SPECTER2 input length (default 160; 256 reads more abstract but embeds ~2× slower) |
| `ATLAS_QUANTIZE=0` | turn off int8 CPU quantisation of SPECTER2 |

### Before a demo

```bash
python precompute.py                     # caches the six demo queries (skips ones already cached)
python precompute.py "your own query"    # cache anything else you plan to show
python check_gemini.py                   # one tiny request: confirms the key, model and billing
```

Open the app ~30 s before presenting: the model load and UMAP compilation happen in the background at startup.

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit page: landing page, immersive map view, session caching, methods panel |
| `atlas_core.py` | the pipeline: OpenAlex, SPECTER2, paradigms, gaps, UMAP, Gemini, payload assembly, caching |
| `map_template.html` | the 3D map: stars, dust, terrain, gap voids, labels, side panels, reading-level slider |
| `static/atlas_vendor.js` | committed bundle of three.js + 3d-force-graph + UnrealBloomPass (rebuild: `cd vendor && npm install && npm run build`) |
| `static/fonts/` | Montserrat (SIL Open Font License), embedded so nothing loads from a CDN |
| `precompute.py`, `check_gemini.py` | demo prep and Gemini diagnostics |
| `.atlas_cache/` | cached maps, Gemini replies and embeddings (git-ignored, safe to delete) |

## Engineering notes

**Latency**
- Startup warm-up (background thread): SPECTER2 load, UMAP's numba JIT on data of the real shape, heavy imports
  (scikit-learn, google-genai), and a keep-alive connection to OpenAlex.
- SPECTER2 runs int8-quantised on CPU with length-sorted batches; embeddings are cached per paper and saved to disk
  off the critical path.
- `NUMBA_THREADING_LAYER=workqueue`: numba's default threading layer contends with torch's OpenMP pool and made
  SPECTER2 8× slower after the first UMAP run.
- UMAP runs in parallel with the 768-D analysis, and Gemini starts the moment paradigms and gaps exist.
- Each map and its rendered HTML are kept in the Streamlit session, so interacting never recomputes anything.
- The front end loads nothing from the network (JS bundle and fonts are inlined), and node positions are fixed from
  UMAP, so the browser runs no force simulation.

**Gemini usage**
- One request per map; the model/settings that last worked are tried first and unknown model names are skipped.
- Replies are cached on disk by prompt, so rebuilding a map (`precompute.py --force`) costs nothing.
- Requests are paced and serialised across sessions. A 429 is retried once only if the server asks for ≤ 20 s;
  otherwise Gemini is skipped until the wait has passed. A 402 (no credits) stops all calls for 10 minutes.
- A failed map is never cached and offers a **Retry Gemini** button instead of retrying on every rerun.

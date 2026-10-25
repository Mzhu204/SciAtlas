"""
SCIATLAS core pipeline.

    OpenAlex search ──► SPECTER2 embeddings (768-D) ──┬─► UMAP 3D layout ─────────────────────────┐
                                                      └─► paradigms (k-means) ─► gaps ─► Gemini ──┴─► map payload

Everything analytical (paradigms, gaps, nearest neighbours) is computed in the ORIGINAL 768-D SPECTER2 space.
UMAP is only used to *draw* the map, so a void on screen is backed by a measured low-density region in embedding
space rather than a projection artefact.

Public entry points:  run(query, n)  · warmup_async() · warmup()  · gemini_key()
"""
from __future__ import annotations

import os

# numba's default (OpenMP/TBB) threading layer fights torch's OpenMP pool and makes SPECTER2 ~8x slower after the
# first UMAP run. "workqueue" avoids that, but it is not thread-safe, so every UMAP call holds _NUMBA_LOCK.
# Must be set before numba (via umap) is imported.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import hashlib
import itertools
import json
import pickle
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")  # sklearn/umap/transformers deprecation chatter is noise in a demo

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".atlas_cache"  # finished maps, Gemini replies and paper embeddings (git-ignored)
CACHE.mkdir(exist_ok=True)

VERSION = "4"  # part of the map cache key; bump only if the payload format changes
LEVELS = ("curious", "student", "expert")  # the three reading levels of every label/summary
STANCES = ("consensus", "supporting", "emerging", "contested", "contrarian")
DEMO_QUERIES = [
    "Does intermittent fasting extend lifespan?",
    "microplastics and human health",
    "alternatives to dark matter",
    "gut microbiome and depression",
    "reasoning in large language models",
    "CRISPR off-target effects",
]


# ======================================================================================== settings
@lru_cache(maxsize=1)
def _secrets() -> dict:
    """.streamlit/secrets.toml (git-ignored), parsed once. Lets CLI scripts share the app's keys."""
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then secrets.toml, then the default. Read at call time, never at import."""
    return os.getenv(name) or _secrets().get(name) or default


def gemini_key() -> str | None:
    return setting("GEMINI_API_KEY") or setting("GOOGLE_API_KEY")


MAX_TOKENS = int(setting("ATLAS_MAX_TOKENS", "160"))  # title + start of abstract; 2x faster than 256 tokens


# ======================================================================================== 1. OpenAlex
OPENALEX = "https://api.openalex.org/works"
_http = requests.Session()  # keep-alive: later requests skip the TCP/TLS handshake
_http.headers["User-Agent"] = "SCIATLAS-hackathon/1.0 (map of science)"


def _clean(s: str | None) -> str:
    """Strip inline HTML (OpenAlex titles contain <i>, <sub> …) and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def _abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}; rebuild the plain text."""
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def _openalex_page(query: str, page: int, per_page: int) -> list[dict]:
    params = {
        "search": query,
        "per_page": per_page,
        "page": page,
        "filter": "has_abstract:true,is_retracted:false",
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,primary_location,authorships",
    }
    if setting("OPENALEX_EMAIL"):
        params["mailto"] = setting("OPENALEX_EMAIL")  # "polite pool": faster, more reliable responses
    if setting("OPENALEX_API_KEY"):
        params["api_key"] = setting("OPENALEX_API_KEY")
    for attempt in range(3):
        try:
            r = _http.get(OPENALEX, params=params, timeout=20)
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code in (429, 500, 502, 503, 504):  # transient: back off and retry
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)
    return []


def fetch_papers(query: str, n: int = 200) -> list[dict]:
    """Up to n relevance-ranked papers that have an abstract, deduplicated by title."""
    query = " ".join(re.sub(r"[?*!:\"()\[\]{}|\\]", " ", query).split())  # OpenAlex treats ?/* as wildcards
    per = min(200, n)  # OpenAlex page size limit
    pages = -(-n // per)
    with ThreadPoolExecutor(pages) as pool:  # pages are fetched concurrently
        results = list(pool.map(lambda p: _openalex_page(query, p, per), range(1, pages + 1)))

    papers, seen = [], set()
    for w in itertools.chain(*results):
        title = _clean(w.get("title"))
        ab = _clean(_abstract(w.get("abstract_inverted_index")))
        key = title.lower()[:90]
        if not title or len(ab) < 80 or key in seen:  # skip stubs and duplicate records
            continue
        seen.add(key)
        src = (w.get("primary_location") or {}).get("source") or {}
        auths = w.get("authorships") or []
        names = [(a.get("author") or {}).get("display_name") for a in auths[:3]]
        papers.append(dict(  # short keys: this ends up in the browser payload
            id=w["id"].rsplit("/", 1)[-1],
            t=title,
            ab=ab,
            yr=w.get("publication_year"),
            c=int(w.get("cited_by_count") or 0),
            v=src.get("display_name") or "",
            au=", ".join(x for x in names if x) + (" et al." if len(auths) > 3 else ""),
            url=w.get("doi") or w["id"],
        ))
    return papers[:n]


# ======================================================================================== 2. SPECTER2
class Specter2:
    """SPECTER2 base + proximity adapter (falls back to the base model if `adapters` fails to load).
    Embeddings are cached per OpenAlex ID, so overlapping queries only embed new papers."""

    BASE = "allenai/specter2_base"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        torch.set_num_threads(os.cpu_count() or 4)
        self.tok = AutoTokenizer.from_pretrained(self.BASE)
        try:
            from adapters import AutoAdapterModel
            model = AutoAdapterModel.from_pretrained(self.BASE)
            model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
            self.variant = "specter2-proximity"
        except Exception as e:  # keep the demo alive even if `adapters` is broken
            print(f"[sciatlas] proximity adapter unavailable ({e}); falling back to specter2_base")
            from transformers import AutoModel
            model = AutoModel.from_pretrained(self.BASE)
            self.variant = "specter2-base"

        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        model.eval().to(self.device)

        # int8 dynamic quantisation: ~2x faster on CPU; cosine 0.9999 vs the full-precision model
        if self.device == "cpu" and setting("ATLAS_QUANTIZE", "1") == "1":
            try:
                q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                self._forward(q, ["warm up" + self.tok.sep_token + "quantised model"])
                model = q
                self.variant += "-int8"
            except Exception as e:
                print(f"[sciatlas] quantisation skipped ({e})")
        self.model = model

        self._lock = threading.Lock()       # one embedding batch at a time (shared across sessions)
        self._save_lock = threading.Lock()  # one cache write at a time
        self._path = CACHE / f"emb_{self.variant}_{MAX_TOKENS}.pkl"
        self._cache: dict[str, np.ndarray] = {}
        if self._path.exists():
            try:
                self._cache = pickle.loads(self._path.read_bytes())
            except Exception:
                self._cache = {}

    def _forward(self, model, texts: list[str]) -> np.ndarray:
        inp = self.tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                       return_tensors="pt", return_token_type_ids=False).to(self.device)
        with self.torch.inference_mode():
            out = model(**inp)
        return out.last_hidden_state[:, 0, :].float().cpu().numpy()  # [CLS] vector, as SPECTER2 prescribes

    def _save(self, snapshot: dict) -> None:
        with self._save_lock:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(snapshot))
            tmp.replace(self._path)  # atomic swap: a crash never leaves a half-written cache

    def embed(self, papers: list[dict]) -> np.ndarray:
        """L2-normalised (n, 768) float32 matrix, one row per paper."""
        with self._lock:
            todo = [p for p in papers if p["id"] not in self._cache]
            if todo:
                sep = self.tok.sep_token
                texts = [p["t"] + sep + p["ab"] for p in todo]  # SPECTER2 input format: title [SEP] abstract
                order = np.argsort([len(t) for t in texts])      # length-sorted batches = minimal padding
                bs = 32 if self.device != "cpu" else 16
                for i in range(0, len(order), bs):
                    idx = order[i:i + bs]
                    for j, v in zip(idx, self._forward(self.model, [texts[j] for j in idx])):
                        self._cache[todo[j]["id"]] = v.astype(np.float16)
                # persist off the critical path (non-daemon, so it still finishes if the process is exiting)
                threading.Thread(target=self._save, args=(dict(self._cache),)).start()
            X = np.stack([self._cache[p["id"]].astype(np.float32) for p in papers])
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


_EMB: Specter2 | None = None
_EMB_LOCK = threading.Lock()
_NUMBA_LOCK = threading.Lock()


def get_embedder() -> Specter2:
    """Process-wide singleton: the model loads once and is shared by every Streamlit session."""
    global _EMB
    with _EMB_LOCK:
        if _EMB is None:
            _EMB = Specter2()
    return _EMB


def warmup() -> None:
    """Pay every one-time cost up front so the first search is as fast as later ones:
    model load, numba JIT for UMAP, slow imports, and the OpenAlex TLS connection. Uses no Gemini tokens."""
    get_embedder()
    try:
        import umap
        Z = np.random.default_rng(0).normal(size=(80, 768)).astype(np.float32)  # same dtype/shape as real data,
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)                           # so the JIT-compiled code is reused
        with _NUMBA_LOCK:
            umap.UMAP(n_components=3, n_neighbors=15, n_epochs=30, metric="cosine", random_state=42).fit_transform(Z)
    except Exception as e:
        print(f"[sciatlas] UMAP warm-up failed ({e})")
    try:  # heavy imports otherwise paid during the first search
        import sklearn.cluster, sklearn.feature_extraction.text, sklearn.metrics  # noqa: F401
        if gemini_key():
            import google.genai  # noqa: F401
    except Exception:
        pass
    try:  # opens the keep-alive connection (TCP + TLS) to OpenAlex
        _http.get(OPENALEX, params={"search": "science", "per_page": 1, "select": "id"}, timeout=5)
    except Exception:
        pass


def warmup_async():
    """warmup() in a background thread (the app calls this once at startup, while the user is still typing)."""
    return ThreadPoolExecutor(1).submit(warmup)


# ======================================================================================== 3. analysis (768-D)
def cluster(X: np.ndarray) -> tuple[np.ndarray, float]:
    """k-means paradigms. k in [5, 9] (fewer for tiny corpora) is chosen by cosine silhouette; clusters smaller than
    3% of the corpus are folded into their nearest neighbour. Returns (labels with 0 = biggest, silhouette)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(X)
    kmin = 5 if n >= 100 else 3
    kmax = max(kmin, min(9, n // 18))
    best = None
    for k in range(kmin, kmax + 1):
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        s = silhouette_score(X, lab, metric="cosine")
        if best is None or s > best[0]:
            best = (s, lab)
    sil, lab = best

    min_size = max(4, int(0.03 * n))
    while True:
        ids, counts = np.unique(lab, return_counts=True)
        if len(ids) <= 3 or counts.min() >= min_size:
            break
        small = ids[np.argmin(counts)]
        cents = {c: X[lab == c].mean(0) for c in ids if c != small}
        for i in np.where(lab == small)[0]:
            lab[i] = max(cents, key=lambda c: X[i] @ cents[c])

    order = np.argsort(-np.bincount(lab))  # relabel so ids are contiguous and 0 is the biggest paradigm
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[lab], float(sil)


def cluster_keywords(papers: list[dict], labels: np.ndarray, K: int, top: int = 4) -> list[list[str]]:
    """Distinctive TF-IDF terms per paradigm: fallback names, and extra context for Gemini."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    try:
        M = vec.fit_transform([p["t"] + ". " + p["ab"] for p in papers])
    except ValueError:  # vocabulary too small (tiny corpus)
        return [["papers"] for _ in range(K)]
    vocab = np.array(vec.get_feature_names_out())
    glob = np.asarray(M.mean(0)).ravel()
    out = []
    for c in range(K):
        score = np.asarray(M[labels == c].mean(0)).ravel() - 0.8 * glob  # distinctive, not merely frequent
        words = []
        for w in vocab[np.argsort(-score)]:
            if not any(w in x or x in w for x in words):  # avoid "plastic" + "microplastic" duplicates
                words.append(w)
            if len(words) == top:
                break
        out.append(words)
    return out


def find_gaps(X: np.ndarray, labels: np.ndarray, k_nn: int = 5, max_gaps: int = 4) -> list[dict]:
    """
    Research gaps = low-density regions *between* neighbouring paradigms, measured in 768-D.

    1. For each paradigm and its 2 most similar paradigms, walk the path between their centroids
       (normalised linear interpolation, i.e. roughly along the unit sphere) in 21 steps.
    2. At every probe point measure the k-NN radius (cosine distance to the k-th nearest paper).
    3. Reference = the same measurement at the two hilltops (centroids). Comparing probe with probe is
       essential: in 768-D an averaged vector always looks denser than a real paper does.
    4. The emptiest point on the bridge is the gap. sparsity = r_kNN(gap) / r_kNN(hilltops);
       density = papers inside a hilltop-sized neighbourhood at the gap / papers inside it at the hilltops.
    Bridges sparser than 1.05x become gaps (the 4 sparsest are kept).
    """
    k = min(k_nn, len(X) - 1)
    K = int(labels.max()) + 1
    Cn = np.stack([X[labels == c].mean(0) for c in range(K)])
    Cn /= np.linalg.norm(Cn, axis=1, keepdims=True)
    CS = Cn @ Cn.T
    np.fill_diagonal(CS, -np.inf)
    pairs = {tuple(sorted((a, int(b)))) for a in range(K) for b in np.argsort(-CS[a])[:2]}

    ts = np.linspace(0.0, 1.0, 21)
    inner = slice(4, 17)  # only look for the gap in the middle of the bridge (t in [0.2, 0.8])
    gaps = []
    for a, b in pairs:
        P = (1 - ts)[:, None] * Cn[a] + ts[:, None] * Cn[b]
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        DP = 1.0 - P @ X.T                                 # cosine distance probe -> every paper
        r = np.partition(DP, k - 1, axis=1)[:, k - 1]      # k-NN radius at each probe
        ref = (r[0] + r[-1]) / 2
        j = inner.start + int(np.argmax(r[inner]))
        top_count = max(((DP[0] < ref).sum() + (DP[-1] < ref).sum()) / 2, 1.0)
        gaps.append(dict(a=int(a), b=int(b), score=round(float(r[j] / ref), 3),
                         density=round(float(min(1.0, (DP[j] < ref).sum() / top_count)), 3),
                         near=[int(i) for i in np.argsort(DP[j])[:4]]))  # closest existing work
    gaps = sorted((g for g in gaps if g["score"] > 1.05), key=lambda g: -g["score"])[:max_gaps]
    for i, g in enumerate(gaps):
        g["id"] = i
    return gaps


# ======================================================================================== 4. 3D layout
def project_3d(X: np.ndarray) -> np.ndarray:
    """UMAP (cosine) to 3D, centred and scaled to roughly ±95 scene units. PCA if UMAP is unavailable."""
    n = len(X)
    try:
        if n < 16:
            raise ValueError("too few points for UMAP")
        import umap
        with _NUMBA_LOCK:
            Y = umap.UMAP(n_components=3, n_neighbors=min(15, n - 1), min_dist=0.3, spread=1.6,
                          metric="cosine", n_epochs=300, random_state=42).fit_transform(X)
    except Exception as e:
        print(f"[sciatlas] UMAP unavailable ({e}); using PCA")
        from sklearn.decomposition import PCA
        Y = PCA(3, random_state=0).fit_transform(X)
    Y = Y - np.median(Y, 0)
    return Y / (np.percentile(np.abs(Y), 98) + 1e-9) * 95


def place_gaps_3d(Y: np.ndarray, gaps: list[dict], clusters: list[dict]) -> None:
    """Draw each (768-D measured) gap at the emptiest visible spot between its two paradigms' 3D centres;
    its radius grows with that empty space and with the measured sparsity."""
    for g in gaps:
        ca, cb = np.array(clusters[g["a"]]["cent"]), np.array(clusters[g["b"]]["cent"])
        P = ca + np.linspace(0.25, 0.75, 25)[:, None] * (cb - ca)
        dmin = np.linalg.norm(P[:, None, :] - Y[None, :, :], axis=2).min(1)
        j = int(np.argmax(dmin))
        g["pos"] = P[j].round(2).tolist()
        g["rad"] = round(float(np.clip(dmin[j] * 0.8, 5, 26) * np.clip(0.6 + 3 * (g["score"] - 1), 0.6, 1.4)), 2)


# ======================================================================================== 5. Gemini
GEMINI_RULES = """
Return ONLY JSON of exactly this shape:
{"consensus":{"answer":{"curious":str,"student":str,"expert":str},
              "strength":"strong|moderate|weak|none",
              "debate":{"curious":str,"student":str,"expert":str}},
 "clusters":[{"id":int,"name":{"curious":str,"student":str,"expert":str},
              "summary":{"curious":str,"student":str,"expert":str},
              "stance":"consensus|supporting|emerging|contested|contrarian"}],
 "contradictions":[{"paper":int,"reason":{"curious":str,"student":str,"expert":str}}],
 "gaps":[{"id":int,"name":{"curious":str,"student":str,"expert":str},
          "question":{"curious":str,"student":str,"expert":str}}]}

Rules
CONSENSUS — this is the most important field.
- consensus.answer is a DIRECT answer to the user's question, in natural language: state the relationship or
  effect that the MAJORITY of the findings above agree on (direction, who/what it applies to, how sure we are).
  Open with the answer itself ("Yes, in animals…", "Most studies find…", "Probably not…").
  Never describe clusters, keywords, paper counts or citation counts. 1-3 sentences per reading level.
- Ground it in the listed findings, not general knowledge. If findings are genuinely split, say that plainly.
- strength: "strong" = most papers agree and the evidence is direct and replicated; "moderate" = a clear majority
  agrees but evidence is limited or indirect (animal, in-vitro, observational); "weak" = a leaning, but sparse or
  inconsistent; "none" = no majority view.
- debate: one sentence per level on the main point that is still disputed or unknown.

STANCE of each cluster, judged ONLY relative to consensus.answer:
- "consensus": the paradigm whose findings most directly establish the answer. Exactly one cluster,
  unless strength is "none", in which case no cluster gets it.
- "supporting": findings agree with or extend the answer.
- "emerging": a newer line of work whose findings do not yet confirm or refute the answer.
- "contested": the cluster's own findings are mixed or conflict with each other about the answer.
- "contrarian": findings mostly dispute the answer.

CONTRADICTIONS: up to 10 individual papers (by [index]) whose OWN finding conflicts with consensus.answer.
  reason: <= 18 words per reading level saying what the paper found and how it conflicts. Only real conflicts; [] is fine.

NAMES: name.curious = 2-4 plain words a 12-year-old understands; name.student = 3-5 words, undergraduate level;
  name.expert = 3-6 words of precise terminology. Distinct across clusters.
SUMMARY: 1-2 sentences per level, written for that reader: what this paradigm found and how that bears on the answer.
GAPS: a short name for the unexplored territory between the two clusters; question = one concrete, answerable
  research question that sits in that gap, phrased for each reading level (the same question, not three different ones).
Exactly one entry per cluster id and per gap id listed above.
"""

# NOTE: the prompt text drives the on-disk reply cache key (llm_*.json). Changing GEMINI_RULES, _finding() or
# _gemini_prompt() means cached replies no longer match and every map costs a fresh request.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _finding(ab: str, limit: int = 180) -> str:
    """The closing sentences of an abstract are where results and conclusions live."""
    s = _SENT.split(ab.strip())
    concl = [x for x in s if re.search(r"\b(conclu|suggest|indicat|demonstrat|found|show|associat|no significant)", x, re.I)]
    text = " ".join(concl[-2:] if concl else s[-2:])
    return text[:limit] + ("…" if len(text) > limit else "")


def _gemini_prompt(query: str, papers: list[dict], labels: np.ndarray, keywords: list[list[str]], gaps: list[dict]) -> str:
    """~5k tokens: per paradigm, the most-cited papers with their key finding, plus the measured gaps."""
    per_cluster = int(setting("GEMINI_PAPERS_PER_CLUSTER", "14"))
    out = [f'The user asked: "{query}"',
           f'Treat it as the question: "What does the research say about {query.rstrip("?")}?"',
           "Below are the most relevant papers, embedded with SPECTER2 and clustered into research paradigms.",
           "Format: [index] (year, citations) title :: key finding from the abstract", ""]
    for c in range(int(labels.max()) + 1):
        idx = sorted(np.where(labels == c)[0], key=lambda i: -papers[i]["c"])
        out.append(f"## Cluster {c}: {len(idx)} papers; distinctive terms: {', '.join(keywords[c])}")
        out += [f"[{i}] ({papers[i]['yr']}, {papers[i]['c']}) {papers[i]['t'][:120]} :: {_finding(papers[i]['ab'])}"
                for i in idx[:per_cluster]]
        out.append("")
    if gaps:
        out.append("## Gaps: measured low-density regions in embedding space between two clusters")
        for g in gaps:
            near = "; ".join(papers[i]["t"][:90] for i in g["near"][:3])
            out.append(f"Gap {g['id']}: between cluster {g['a']} and cluster {g['b']}. Closest papers: {near}")
    out.append(GEMINI_RULES)
    return "\n".join(out)


# --- request policy: every Streamlit session shares one process, so this state is global.
#     One request at a time, paced; remember the model/settings that work; back off completely on 429/402.
GEMINI_MAX_WAIT = 20  # honour a server "retry in Ns" only if it is this short; otherwise fail fast
_GEM_LOCK = threading.Lock()
_GEM = {"last": 0.0, "cooldown_until": 0.0, "working": None, "dead": set()}


def _gemini_models() -> list[str]:
    return list(dict.fromkeys([setting("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-flash-latest"]))


def _is_billing(msg: str) -> bool:
    # 402 "prepayment credits are depleted" also says RESOURCE_EXHAUSTED, but waiting will never fix it
    return "402" in msg or "credits are depleted" in msg or "billing" in msg.lower()


def _is_rate_limit(msg: str) -> bool:
    return not _is_billing(msg) and ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower())


def _retry_after(msg: str) -> float | None:
    m = re.search(r"retry(?:Delay)?\D{0,12}?(\d+(?:\.\d+)?)\s*s", msg, re.I)
    return float(m.group(1)) if m else None


def gemini_generate(prompt: str, max_tokens: int = 4096) -> tuple[str, str]:
    """One JSON completion, as cheaply as possible. Returns (text, model) or raises the last error.

    Order of attempts: the last (model, thinking-off?) combination that worked, then each configured model with
    thinking disabled, then with default thinking (only if the model rejected the thinking config)."""
    from google import genai
    from google.genai import types

    min_interval = float(setting("GEMINI_MIN_INTERVAL", "6"))  # seconds between requests (free tier ~10 RPM)
    with _GEM_LOCK:
        wait = _GEM["cooldown_until"] - time.time()
        if wait > 0:
            raise RuntimeError(f"Gemini is rate-limited; skipping calls for another {wait:.0f}s")
        client = genai.Client(api_key=gemini_key(), http_options=types.HttpOptions(timeout=45_000))
        combos = ([_GEM["working"]] if _GEM["working"] else []) + [(m, t) for m in _gemini_models() for t in (True, False)]
        last = None
        for model, thinking_off in dict.fromkeys(combos):
            if model in _GEM["dead"]:
                continue
            cfg = dict(response_mime_type="application/json", temperature=0.3, max_output_tokens=max_tokens)
            if thinking_off:  # fewer tokens and lower latency
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            for attempt in range(2):
                pause = _GEM["last"] + min_interval - time.time()
                if pause > 0:
                    time.sleep(pause)
                _GEM["last"] = time.time()
                try:
                    resp = client.models.generate_content(model=model, contents=prompt,
                                                          config=types.GenerateContentConfig(**cfg))
                    _GEM["working"] = (model, thinking_off)
                    return resp.text or "", model
                except Exception as e:
                    last, msg = e, str(e)
                    if _is_billing(msg):
                        _GEM["cooldown_until"] = time.time() + 600  # same project for every model: stop
                        raise
                    if not _is_rate_limit(msg):
                        break
                    delay = _retry_after(msg) or 60.0
                    if attempt == 0 and delay <= GEMINI_MAX_WAIT:
                        time.sleep(delay)  # short server-requested wait: retry the same call once
                        continue
                    _GEM["cooldown_until"] = time.time() + delay
                    raise
            msg = str(last)
            if "404" in msg or "not found" in msg.lower():
                _GEM["dead"].add(model)  # unknown model name: never try it again this process
            elif "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                raise last  # no model will work with a bad key
            elif not ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
                raise last  # only a rejected thinking config is worth retrying with the other variant
            if _GEM["working"] == (model, thinking_off):
                _GEM["working"] = None
        raise last or RuntimeError("no usable Gemini model")


def gemini_label(query, papers, labels, keywords, gaps) -> tuple[dict | None, str, str]:
    """Returns (parsed JSON or None, model used / status, error message). Replies are cached on disk by prompt."""
    key = gemini_key()
    if not key:
        return None, "none (no GEMINI_API_KEY)", ""
    prompt = _gemini_prompt(query, papers, labels, keywords, gaps)
    path = CACHE / f"llm_{hashlib.sha1(prompt.encode()).hexdigest()[:20]}.json"
    if path.exists():
        cached = json.loads(path.read_text("utf-8"))
        return cached["data"], cached["model"] + " (cached)", ""
    try:
        txt, model = gemini_generate(prompt)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip())
        path.write_text(json.dumps({"model": model, "data": data}, ensure_ascii=False), "utf-8")
        return data, model, ""
    except Exception as e:
        err = str(e).replace(key, "<key>")[:400]
        print(f"[sciatlas] Gemini failed: {err}")
        return None, f"fallback ({type(e).__name__})", err


# ======================================================================================== 6. payload
def _levels(obj, fallback: str) -> dict:
    """Normalise a Gemini text field into {curious, student, expert}, filling any missing level."""
    if isinstance(obj, str) and obj.strip():
        return {lv: obj.strip() for lv in LEVELS}
    obj = obj if isinstance(obj, dict) else {}
    vals = {lv: str(obj.get(lv) or "").strip() for lv in LEVELS}
    any_val = next((v for v in (vals["student"], vals["expert"], vals["curious"]) if v), fallback)
    return {lv: vals[lv] or any_val for lv in LEVELS}


def _by_id(items: list, item: dict):
    """items[int(item['id'])] or None, tolerating string ids and out-of-range values from the model."""
    try:
        return items[int(item["id"])]
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def assemble(query, papers, X, Y, labels, sil, keywords, gaps, llm, meta) -> dict:
    """Merge geometry, analysis and (optional) Gemini output into the JSON the browser renders.
    Without Gemini nothing claims agreement: stances are "unrated" and there are no challengers."""
    K = int(labels.max()) + 1
    cites = np.array([p["c"] for p in papers], float)
    lc = np.log1p(cites) / max(np.log1p(cites.max()), 1e-9)  # 0..1 log-citations

    clusters = []
    for c in range(K):
        m = labels == c
        P = Y[m]
        cent = np.median(P, 0)
        rad = float(np.percentile(np.linalg.norm(P - cent, axis=1), 70)) if m.sum() > 1 else 6.0
        idx = sorted(np.where(m)[0], key=lambda i: -papers[i]["c"])
        kw = " · ".join(keywords[c][:2]).title()
        clusters.append(dict(
            id=c, size=int(m.sum()), cent=cent.round(2).tolist(), rad=round(max(rad, 6.0), 2),
            keywords=keywords[c], top=[papers[i]["id"] for i in idx[:6]],
            name={lv: kw for lv in LEVELS},
            summary={lv: f"{int(m.sum())} papers centred on {', '.join(keywords[c][:3])}." for lv in LEVELS},
            stance="unrated",
            _cites=float(cites[m].sum()),
        ))

    place_gaps_3d(Y, gaps, clusters)
    for g in gaps:
        A, B = clusters[g["a"]]["name"]["student"], clusters[g["b"]]["name"]["student"]
        g["name"] = {lv: f"{A} × {B}" for lv in LEVELS}
        q = f"What connects {A.lower()} and {B.lower()}? Almost no papers sit between them."
        g["question"] = {lv: q for lv in LEVELS}
        g["near"] = [papers[i]["id"] for i in g["near"]]

    consensus = dict(available=False, answer={lv: "" for lv in LEVELS}, debate={lv: "" for lv in LEVELS},
                     strength="none")
    contras: list[dict] = []

    if llm:
        cons = llm.get("consensus") if isinstance(llm.get("consensus"), dict) else {}
        strength = cons.get("strength")
        consensus = dict(
            available=True,
            answer=_levels(cons.get("answer") or cons, "The papers mapped here do not converge on a single answer."),
            debate=_levels(cons.get("debate"), ""),
            strength=strength if strength in ("strong", "moderate", "weak", "none") else "moderate",
        )
        for item in llm.get("clusters") or []:
            c = _by_id(clusters, item)
            if c:
                c["name"] = _levels(item.get("name"), c["name"]["student"])
                c["summary"] = _levels(item.get("summary"), c["summary"]["student"])
                c["stance"] = item.get("stance") if item.get("stance") in STANCES else "supporting"
        # exactly one "consensus" paradigm (the most-cited claimant), or none if there is no clear consensus
        claimants = sorted((c for c in clusters if c["stance"] == "consensus"), key=lambda c: -c["_cites"])
        for c in claimants[1:]:
            c["stance"] = "supporting"
        if consensus["strength"] == "none":
            if claimants:
                claimants[0]["stance"] = "contested"
        elif not claimants:
            max(clusters, key=lambda c: c["_cites"])["stance"] = "consensus"
        for item in llm.get("gaps") or []:
            g = _by_id(gaps, item)
            if g:
                g["name"] = _levels(item.get("name"), g["name"]["student"])
                g["question"] = _levels(item.get("question"), g["question"]["student"])
        for item in llm.get("contradictions") or []:
            try:
                i = int(item["paper"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= i < len(papers) and len(contras) < 10:
                contras.append(dict(paper=i, reason=_levels(item.get("reason"), "Reports findings that conflict with the consensus.")))

    for c in clusters:
        del c["_cites"]
    why = {d["paper"]: d["reason"] for d in contras}

    # share of the mapped literature sitting in agreeing / mixed / disputing paradigms (the panel's bar)
    share = lambda sts: round(100 * sum(c["size"] for c in clusters if c["stance"] in sts) / len(papers))
    consensus.update(agree=share(("consensus", "supporting")), mixed=share(("contested", "emerging")),
                     dispute=share(("contrarian",)), challengers=len(contras))

    nodes = [dict(
        id=p["id"], t=p["t"], ab=p["ab"][:900] + ("…" if len(p["ab"]) > 900 else ""),
        yr=p["yr"], c=p["c"], v=p["v"], au=p["au"], url=p["url"], cl=int(labels[i]),
        x=round(float(Y[i, 0]), 2), y=round(float(Y[i, 1]), 2), z=round(float(Y[i, 2]), 2),
        r=round(1.3 + 4.2 * float(lc[i]) ** 0.8, 2),   # citation size factor (stars)
        w=round(0.35 + 0.65 * float(lc[i]), 3),        # citation weight (terrain height)
        contra=i in why, why=why.get(i, ""),
    ) for i, p in enumerate(papers)]

    # faint "semantic threads": each paper to its 2 nearest neighbours in 768-D
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :2]
    pairs = sorted({tuple(sorted((i, int(j)))) for i in range(len(papers)) for j in nn[i]})
    links = [[papers[a]["id"], papers[b]["id"]] for a, b in pairs]

    meta.update(query=query, n=len(papers), k=K, silhouette=round(sil, 3))
    return dict(meta=meta, consensus=consensus, clusters=clusters, gaps=gaps, nodes=nodes, links=links)


# ======================================================================================== 7. pipeline
def _cache_path(query: str, n: int) -> Path:
    # whether a Gemini key is configured is part of the key: adding one later must not serve the unlabelled map
    llm_tag = "llm" if gemini_key() else "nollm"
    return CACHE / f"map_{hashlib.sha1(f'{VERSION}|{query.lower()}|{n}|{llm_tag}'.encode()).hexdigest()[:16]}.json"


def run(query: str, n: int = 200, on_step=None, use_cache: bool = True) -> dict:
    """Full pipeline for one search. on_step(str) receives progress messages for the UI."""
    query = " ".join(query.split())
    step = on_step or (lambda s: None)
    path = _cache_path(query, n)
    if use_cache and path.exists():
        step("Loaded from the cache")
        data = json.loads(path.read_text("utf-8"))
        data["meta"]["cached"] = True
        return data

    emb = get_embedder()  # one-time model load (normally pre-warmed) is not counted as query latency
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def timed(name, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        timings[name] = round(time.perf_counter() - t, 2)
        return out

    step("Searching OpenAlex…")
    papers = timed("OpenAlex fetch", fetch_papers, query, n)
    if len(papers) < 12:
        raise ValueError(f"OpenAlex returned only {len(papers)} papers with abstracts. Try a broader query.")

    step(f"Embedding {len(papers)} papers with SPECTER2…")
    X = timed("SPECTER2 embed", emb.embed, papers)
    # drop off-topic search hits (far from the corpus centroid) so they don't form fake paradigms
    mu = X.mean(0)
    s = X @ (mu / np.linalg.norm(mu))
    keep = s >= s.mean() - 2.5 * s.std()
    papers, X = [p for p, k in zip(papers, keep) if k], X[keep]

    # The 3D layout needs only X, so it runs alongside the 768-D analysis; Gemini starts as soon as the
    # paradigms and gaps it describes exist. Shut down without waiting so a Gemini timeout returns immediately
    # (the worker still writes its reply to the llm_ cache, so a retry costs nothing).
    pool = ThreadPoolExecutor(2)
    try:
        layout = pool.submit(timed, "UMAP 3D (parallel)", project_3d, X)

        step("Finding paradigms and research gaps in 768-D…")
        t = time.perf_counter()
        labels, sil = cluster(X)
        keywords = cluster_keywords(papers, labels, int(labels.max()) + 1)
        gaps = find_gaps(X, labels)
        timings["Paradigms + gaps (768-D)"] = round(time.perf_counter() - t, 2)

        step("Gemini is reading the findings…" if gemini_key() else "Drawing the map…")
        t = time.perf_counter()
        labelling = pool.submit(gemini_label, query, papers, labels, keywords, gaps)
        Y = layout.result()
        try:
            llm, model_used, llm_error = labelling.result(timeout=90)
        except Exception as e:
            llm, model_used = None, f"fallback ({type(e).__name__})"
            llm_error = (str(e) or f"{type(e).__name__}: Gemini did not answer within 90s")[:400]
        timings["Gemini"] = round(time.perf_counter() - t, 2)
    finally:
        pool.shutdown(wait=False)

    meta = dict(timings=timings, embedder=emb.variant, llm=model_used, llm_error=llm_error, cached=False)
    data = assemble(query, papers, X, Y, labels, sil, keywords, gaps, llm, meta)
    data["meta"]["total"] = round(time.perf_counter() - t0, 2)
    if not llm_error and not model_used.startswith("fallback"):  # never cache a map whose Gemini call failed
        path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data

"""
SCIATLAS: a navigable 3D map of scientific research.        Run:  streamlit run app.py

Page flow
  landing page (big header, demo chips, how-it-works cards)
      └─ after a search ─► immersive view: one slim control row + the 3D map filling the window
                           (the map itself can go true full screen with its ⛶ button or the F key)

The heavy lifting lives in atlas_core.py; the 3D map is map_template.html rendered inside an iframe.
API keys are read by atlas_core from env vars or .streamlit/secrets.toml.
"""
import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import atlas_core as core

st.set_page_config(page_title="SCIATLAS · the shape of science", page_icon="🌌", layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def start_warmup():
    """Once per server process: load SPECTER2, JIT-compile UMAP, pre-import libraries and open the OpenAlex
    connection in the background, so the first search is as fast as later ones."""
    return core.warmup_async()


start_warmup()

HERE = Path(__file__).parent


# ------------------------------------------------------------------------------------------- static assets
@st.cache_resource(show_spinner=False)
def font_css() -> str:
    """Montserrat as data URLs (no font CDN at demo time). It is a variable font (weights 100-900), so Regular 400
    and Medium 500 come from the same files. Subsets/unicode-ranges as served by Google Fonts (SIL OFL)."""
    subsets = {
        "montserrat-latin-ext.woff2": "U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, "
                                      "U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, "
                                      "U+2113, U+2C60-2C7F, U+A720-A7FF",
        "montserrat-latin.woff2": "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, "
                                  "U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD",
    }
    faces = []
    for fname, unicode_range in subsets.items():
        b64 = base64.b64encode((HERE / "static" / "fonts" / fname).read_bytes()).decode()
        faces.append(f"@font-face{{font-family:'Montserrat';font-style:normal;font-weight:100 900;font-display:block;"
                     f"src:url(data:font/woff2;base64,{b64}) format('woff2');unicode-range:{unicode_range};}}")
    return "\n".join(faces)


@st.cache_resource(show_spinner=False)
def map_template() -> str:
    """map_template.html with fonts and the JS bundle (three.js + 3d-force-graph + bloom) inlined, so the map
    makes no CDN requests and works offline. Only the __DATA__ placeholder is left to fill per map."""
    vendor = (HERE / "static" / "atlas_vendor.js").read_text("utf-8").replace("</script", "<\\/script")
    return ((HERE / "map_template.html").read_text("utf-8")
            .replace("__FONTS__", font_css()).replace("__VENDOR__", vendor))


def render_map(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")  # never close the <script> early
    return map_template().replace("__DATA__", data)


# ------------------------------------------------------------------------------------------- page styling
# palette: #D9FCED mint · #09BBC8 teal · #405FFA blue · #3B3664 dusk · #07240F forest
# fonts:   Montserrat Medium (500) for headings/labels, Montserrat Regular (400) for body text
# NOTE: st.html sanitises its input, so never put anything tag-like (e.g. "<height>") inside these CSS strings.
PAGE_CSS = """
#MainMenu, header, footer, [data-testid="stToolbar"] {visibility:hidden; height:0;}
html, body, .stApp, .stApp button, .stApp input, .stApp textarea, .stApp select, [data-testid="stMarkdownContainer"],
[data-baseweb="select"] * {font-family:'Montserrat', sans-serif !important;}
.stApp {font-weight:400; color:#D9FCED;
  background: radial-gradient(1100px 560px at 50% -14%, rgba(7,36,15,.95) 0%, rgba(59,54,100,.22) 45%, #040908 78%);}
.block-container {padding-top:1.1rem; padding-bottom:.5rem; max-width:100%;}
h1, h2, h3, h4, b, strong, [data-testid="stMetricValue"] {font-weight:500 !important;}
.sciatlas-h {display:flex; align-items:baseline; gap:20px; margin-bottom:.35rem;}
.sciatlas-title {font-size:2rem; font-weight:500; letter-spacing:.42em; margin:0; color:#D9FCED;}
.sciatlas-sub {color:rgba(217,252,237,.55); font-size:.92rem; letter-spacing:.02em;}
.sciatlas-title span {color:#09BBC8;}
div[data-testid="stTextInput"] input {background:rgba(4,9,8,.7); border:1px solid rgba(217,252,237,.16);
  border-radius:8px; font-size:1rem; padding:.7rem 1rem; color:#D9FCED;}
div[data-testid="stTextInput"] input:focus {border-color:#09BBC8; box-shadow:0 0 0 1px #09BBC8;}
div[data-testid="stFormSubmitButton"] button {background:#09BBC8; border:0; border-radius:8px; font-weight:500;
  letter-spacing:.14em; text-transform:uppercase; font-size:.78rem; height:2.95rem; color:#040908;}
div[data-testid="stFormSubmitButton"] button:hover {background:#405FFA; color:#D9FCED;}
div[data-testid="stButton"] button {border-radius:999px; background:transparent; border:1px solid rgba(217,252,237,.14);
  color:rgba(217,252,237,.62); font-size:.76rem; padding:.15rem .8rem; min-height:0;}
div[data-testid="stButton"] button:hover {border-color:#09BBC8; color:#D9FCED;}
iframe {border-radius:12px; border:1px solid rgba(217,252,237,.08);}
.card {background:rgba(59,54,100,.14); border:1px solid rgba(217,252,237,.09); border-radius:12px; padding:18px; height:100%;}
.card h4 {margin:.15rem 0 .45rem; color:#D9FCED; font-weight:500; letter-spacing:.04em;}
.card p {color:rgba(217,252,237,.55); font-size:.86rem; line-height:1.55; margin:0;}
.card .n {font-size:.7rem; letter-spacing:.24em; color:#09BBC8; font-weight:500;}
"""

# Immersive view only: compact controls, and the map iframe stretched to the bottom of the window.
# Streamlit pins the iframe's wrapper with a fixed flex-basis, so that has to be overridden as well as the height.
IMMERSIVE_CSS = """
.block-container {padding-top:.55rem !important; padding-bottom:0 !important;}
iframe[data-testid="stIFrame"] {height:calc(100vh - 82px) !important; min-height:520px;}
div[data-testid="stElementContainer"]:has(> iframe[data-testid="stIFrame"]) {
  height:calc(100vh - 82px) !important; flex:0 0 calc(100vh - 82px) !important; min-height:520px;}
div[data-testid="stTextInput"] input {padding:.45rem .9rem; font-size:.92rem;}
div[data-testid="stFormSubmitButton"] button {height:2.45rem;}
div[data-testid="stPopover"] button {height:2.45rem; border-radius:8px;}
/* the SCIATLAS logo is a button (key="home") that returns to the landing page; strip all button chrome */
.st-key-home button {background:transparent !important; border:0 !important; box-shadow:none !important;
  padding:0 !important; min-height:0 !important; justify-content:flex-start;}
.st-key-home button p {font-size:1.05rem; font-weight:500; letter-spacing:.34em; white-space:nowrap; color:#D9FCED;
  transition:opacity .15s;}
.st-key-home button strong {color:#09BBC8; font-weight:500;}
.st-key-home button:hover p {opacity:.75;}
"""

st.html(f"<style>{font_css()}\n{PAGE_CSS}</style>")


# ------------------------------------------------------------------------------------------- search state
if "qbox" not in st.session_state:  # first visit: a ?q= link opens straight into that map
    st.session_state.qbox = st.query_params.get("q", "")
    st.session_state.q = st.session_state.qbox


def pick(q: str):
    """Demo chip / Examples entry clicked."""
    st.session_state.qbox = q
    st.session_state.q = q


def go_home():
    """Logo clicked: back to the landing page. Finished maps stay in the session, so reopening one is instant."""
    st.session_state.q = ""
    st.session_state.qbox = ""
    st.query_params.clear()


def submit():
    """Runs before the script reruns, so the layout already knows whether to switch to the immersive view."""
    st.session_state.q = st.session_state.qbox.strip()


def search_form(compact: bool) -> int:
    """Search box + paper count + submit. Returns the selected number of papers."""
    with st.form("search", border=False):
        c1, c2, c3 = st.columns([6, 1.4, 1.25] if compact else [8, 1.25, 1.35], vertical_alignment="center")
        c1.text_input("Search", key="qbox", label_visibility="collapsed",
                      placeholder="e.g.  Does intermittent fasting extend lifespan?")
        n = c2.selectbox("Papers", [100, 150, 200, 300], index=2, key="n", label_visibility="collapsed",
                         format_func=lambda v: f"{v} papers")
        c3.form_submit_button("Map it  ✦", on_click=submit, width="stretch")
    return int(n)


q = st.session_state.get("q", "").strip()

# ------------------------------------------------------------------------------------------- landing page
if not q:
    st.markdown('<div class="sciatlas-h"><div class="sciatlas-title">SCI<span>ATLAS</span></div>'
                '<div class="sciatlas-sub">Search a field or ask a question and explore its research landscape in 3D.</div></div>',
                unsafe_allow_html=True)
    search_form(compact=False)
    for col, dq in zip(st.columns(len(core.DEMO_QUERIES)), core.DEMO_QUERIES):
        col.button(dq, key=f"demo_{dq}", on_click=pick, args=(dq,), width="stretch")
    st.write("")
    steps = [
        ("01", "Retrieve", "Up to 300 of the most relevant papers with abstracts are pulled from OpenAlex."),
        ("02", "Embed", "SPECTER2 with its proximity adapter turns every paper into a 768-D vector, so papers on similar science land near each other."),
        ("03", "Measure", "Paradigms and research gaps are computed in the full 768-D space. A void on the map is a measured void in the literature."),
        ("04", "Render", "UMAP projects to 3D, Gemini states the consensus and names every paradigm at three reading levels, and the map renders as a galaxy."),
    ]
    for col, (num, h, p) in zip(st.columns(4), steps):
        col.markdown(f'<div class="card"><div class="n">{num}</div><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------------------------------- immersive map view
st.html(f"<style>{IMMERSIVE_CSS}</style>")
logo, mid, examples = st.columns([1.15, 7.7, 1.0], vertical_alignment="center")
logo.button("SCI**ATLAS**", key="home", on_click=go_home, help="Back to the start page")
with mid:
    n = search_form(compact=True)
with examples.popover("Examples"):
    for dq in core.DEMO_QUERIES:
        st.button(dq, key=f"demo_{dq}", on_click=pick, args=(dq,), width="stretch")

st.query_params["q"] = q  # shareable link

# Every widget interaction reruns this script. Each finished map (and its rendered HTML) is kept in the session,
# so a rerun never repeats the pipeline, the Gemini call, or even the HTML build. Only a new query or an
# explicit retry does.
maps = st.session_state.setdefault("maps", {})
map_key = (q.lower(), n)
if map_key not in maps:
    slot = st.empty()  # progress shows while mapping, then disappears so it doesn't take map space
    try:
        with slot.container(), st.status(f"Mapping **{q}**…", expanded=False) as status:
            result = core.run(q, n, on_step=lambda s: status.update(label=s))
    except Exception as e:
        slot.empty()
        st.error(f"Couldn't map that query: {e}")
        st.stop()
    slot.empty()
    meta = result["meta"]
    st.toast(f"Loaded **{q}** from cache" if meta.get("cached")
             else f"Mapped {meta['n']} papers in {meta['total']:.1f}s", icon="🌌")
    maps[map_key] = {"payload": result, "html": render_map(result)}
payload, html = maps[map_key]["payload"], maps[map_key]["html"]
meta = payload["meta"]

if meta.get("llm_error"):
    w1, w2 = st.columns([8, 1.35])
    w1.warning("Gemini failed, so this map has no consensus answer, stances or challengers. "
               f"Run `python check_gemini.py` to diagnose.  \nError: `{meta['llm_error']}`")
    if w2.button("Retry Gemini", width="stretch"):
        maps.pop(map_key, None)  # embeddings are cached, so a retry costs ~1 Gemini request + a few seconds
        st.rerun()
elif not payload["consensus"].get("available"):
    st.caption("No GEMINI_API_KEY set: showing structure and gaps only (no consensus answer, stances or challengers).")

components.html(html, height=820, scrolling=False)  # height is overridden by IMMERSIVE_CSS to fill the window

# ------------------------------------------------------------------------------------------- methods (below the fold)
with st.expander("🔬 Under the hood: how the map is built"):
    a, b, c, d, e = st.columns(5)
    a.metric("Papers", meta["n"])
    b.metric("Paradigms (k)", meta["k"])
    c.metric("Gaps found", len(payload["gaps"]))
    d.metric("Silhouette (cosine)", meta["silhouette"])
    e.metric("Pipeline time", f"{meta.get('total', 0):.1f}s")
    st.caption(f"Embedder: `{meta['embedder']}` · Labeller: `{meta['llm']}`"
               + (" · served from cache" if meta.get("cached") else ""))
    if meta.get("timings"):
        st.dataframe([{"stage": k, "seconds": v} for k, v in meta["timings"].items()], hide_index=True)
    if payload["gaps"]:
        names = {cl["id"]: cl["name"]["student"] for cl in payload["clusters"]}
        st.dataframe([{"gap": g["name"]["student"], "between": f"{names[g['a']]} ↔ {names[g['b']]}",
                       "sparsity (kNN radius vs hilltops)": g["score"],
                       "density (% of hilltops)": round(100 * g["density"])} for g in payload["gaps"]],
                     hide_index=True)
    st.markdown("""
- **Paradigms**: k-means on L2-normalised SPECTER2 vectors (768-D), with *k* ∈ [5, 9] picked by cosine silhouette.
- **Gaps**: for each paradigm and its two most similar neighbours, we walk the path between their 768-D centroids and
  measure the k-NN radius (k = 5) at each step. **Sparsity** is the emptiest point's radius compared with the same
  measurement at the two hilltops (the centroids). **Density** is how many papers sit inside a hilltop-sized
  neighbourhood there, compared with the hilltops. Gaps are measured before any dimensionality reduction; UMAP only
  decides where on screen they are drawn.
- **Layout**: UMAP (cosine, 15 neighbours) to 3D. Each paper is a star: size from citations, brightness from how
  crowded its neighbourhood is. The terrain is a citation-weighted density field, carved out wherever a gap was measured.
- **Consensus**: Gemini reads each paradigm's most-cited papers (title and key finding, i.e. the concluding sentences
  of the abstract) and writes a direct answer to your question stating what the majority of findings agree on, with a
  strength rating (strong / moderate / weak / none) and the main open debate.
- **Stances and challengers** are judged against that stated answer. Each paradigm is rated *supporting*, *emerging*,
  *contested* or *contrarian* (or *consensus* for the one that establishes it). Individual papers whose own findings
  conflict with it are ringed as challengers. The agreement bar is the share of papers in each kind of paradigm.
""")

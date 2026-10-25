"""Warm the cache before a demo so every map opens instantly.

    python precompute.py                      # all demo queries (skips ones already cached)
    python precompute.py "your query" ...     # specific queries
    python precompute.py --force ...          # rebuild even if cached (unchanged Gemini replies are reused for free)

Gemini requests are paced by atlas_core (GEMINI_MIN_INTERVAL), so a batch never trips the per-minute limit.
"""
import sys
import time

import atlas_core as core

if __name__ == "__main__":
    force = "--force" in sys.argv
    queries = [a for a in sys.argv[1:] if a != "--force"] or core.DEMO_QUERIES
    core.warmup()
    for q in queries:
        t = time.perf_counter()
        d = core.run(q, 200, on_step=lambda s: print("   ", s), use_cache=not force)
        m = d["meta"]
        ok = not m.get("llm_error")
        status = m["llm"] if ok else f"GEMINI FAILED: {m['llm_error'][:120]}"
        print(f"{'✓' if ok else '✗'} {q!r}: {m['n']} papers, {m['k']} paradigms, {len(d['gaps'])} gaps, "
              f"{sum(x['contra'] for x in d['nodes'])} challengers, {time.perf_counter() - t:.1f}s  [{status}]")

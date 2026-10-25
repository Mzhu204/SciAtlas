"""Diagnose Gemini access:  python check_gemini.py
Uses at most one tiny request per model and stops at the first success, so it barely touches your quota."""
import atlas_core as core

key = core.gemini_key()
if not key:
    raise SystemExit("No key found. Set GEMINI_API_KEY or add it to .streamlit/secrets.toml")
print(f"key: ...{key[-4:]}  (length {len(key)})")

from google import genai

client = genai.Client(api_key=key)
try:  # listing models does not count against generation quota
    names = [m.name.removeprefix("models/") for m in client.models.list() if "flash" in m.name]
    print("flash models your key can see:", ", ".join(names) or "none")
except Exception as e:
    print("could not list models:", str(e).replace(key, "<key>")[:300])

try:
    txt, model = core.gemini_generate('Reply with JSON {"ok": true}', max_tokens=32)
    print(f"OK    {model}: {txt.strip()[:60]}")
except Exception as e:
    msg = str(e).replace(key, "<key>")
    print(f"FAIL  {msg[:400]}")
    if core._is_billing(msg):
        print("\nBilling problem, not a rate limit: the project behind this key has no credits left. Top up or switch"
              " to the free tier at https://ai.studio/projects, or use a key from a project with credits.")
    elif core._is_rate_limit(msg):
        print("\nRate limited. Wait for the quota window to reset (per-minute limits reset within ~60s; daily limits"
              " at midnight Pacific), or set GEMINI_MODEL to a model with its own quota.")

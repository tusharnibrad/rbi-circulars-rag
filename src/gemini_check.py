"""Diagnose why Gemini answers take minutes. Needs no search models, so it starts instantly.

Run:  python src\\gemini_check.py
It makes 4 timed calls (each gives up after 90 s, so you never wait 5 minutes):
  1. tiny prompt, default settings          -> is the API / network / free tier itself slow?
  2. tiny prompt, thinking_level="low"      -> does less "thinking" change the speed?
  3. real RAG-sized prompt, default         -> is it the long prompt?
  4. real RAG-sized prompt, thinking low    -> the likely fix
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from answer import DEFAULT_MODEL, SYSTEM_PROMPT, build_user_prompt

load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
TIMEOUT_MS = 90_000


def real_prompt():
    """A prompt as big as the app sends: the first 5 digital-lending chunks, no search models needed."""
    chunks = [json.loads(l) for l in Path("data/processed/chunks.jsonl").open(encoding="utf-8")]
    picked = [c for c in chunks if c["doc_id"] == "digital_lending_2025"][8:13]
    return build_user_prompt("What is the cooling-off period for a digital loan?", picked)


def call(client, label, prompt, system=None, level=None):
    cfg = {"temperature": 0.1}
    if system:
        cfg["system_instruction"] = system
    if level:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    t0 = time.time()
    try:
        r = client.models.generate_content(model=MODEL, contents=prompt,
                                           config=types.GenerateContentConfig(**cfg))
        u = r.usage_metadata
        print(f"{label:34} {time.time() - t0:6.1f} s | prompt tokens={getattr(u, 'prompt_token_count', '?')} "
              f"thinking tokens={getattr(u, 'thoughts_token_count', '?')} "
              f"output tokens={getattr(u, 'candidates_token_count', '?')}")
    except Exception as e:
        print(f"{label:34} {time.time() - t0:6.1f} s | FAILED: {str(e)[:160]}")


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    print(f"model = {MODEL}\n")
    tiny = "Reply with the single word: OK"
    big = real_prompt()
    call(client, "1. tiny, default", tiny)
    call(client, "2. tiny, thinking low", tiny, level="low")
    call(client, "3. real prompt, default", big, SYSTEM_PROMPT)
    call(client, "4. real prompt, thinking low", big, SYSTEM_PROMPT, level="low")


if __name__ == "__main__":
    main()

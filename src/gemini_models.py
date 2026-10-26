"""Find a Gemini model that answers fast on YOUR key.

Run:  python src\\gemini_models.py
Lists the "flash" models your key can use and times a tiny call on each (30 s limit per model).
Then put the fastest model's name in .env:   GEMINI_MODEL=<name>
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
TIMEOUT_MS = 30_000
MAX_MODELS = 8


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))

    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = m.name.replace("models/", "")
        if "flash" in name and "generateContent" in actions and "image" not in name and "tts" not in name:
            names.append(name)
    names = sorted(set(names), reverse=True)[:MAX_MODELS]
    print(f"Testing {len(names)} flash models (30 s limit each):\n")

    for name in names:
        t0 = time.time()
        try:
            r = client.models.generate_content(model=name, contents="Reply with the single word: OK",
                                               config=types.GenerateContentConfig(temperature=0.1))
            print(f"  {name:38} {time.time() - t0:5.1f} s   OK  -> {r.text.strip()[:20]!r}")
        except Exception as e:
            print(f"  {name:38} {time.time() - t0:5.1f} s   FAILED: {str(e)[:90]}")
    print("\nPick the fastest OK one and set  GEMINI_MODEL=<name>  in .env")


if __name__ == "__main__":
    main()

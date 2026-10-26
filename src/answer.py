"""Step 5 of the pipeline: question -> grounded answer with citations.

Flow: retrieve top-k chunks -> put them in a numbered CONTEXT block -> ask Gemini to answer
ONLY from that context and cite chunk numbers like [1], [2] -> print answer + source list.

Run:
  python src\\answer.py "What is the cooling-off period for a digital loan?"
  python src\\answer.py "What is the agriculture target under PSL?" --active-only
  python src\\answer.py "What are the V-CIP requirements for small finance banks?" --k 6

.env needs:   GEMINI_API_KEY=...
              GEMINI_MODELS=gemini-flash-lite-latest,gemini-3.8-flash   (tried in this order; first one that answers wins)
"""
import argparse
import os
import time

from dotenv import load_dotenv

from retrieve import Retriever

load_dotenv()
DEFAULT_MODEL = "gemini-flash-lite-latest"   # found with src\\gemini_models.py; the big flash models were busy (503/504)
REFUSAL = "I could not find this in the provided RBI documents."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about Reserve Bank of India (RBI) regulations.

Rules:
1. Use ONLY the numbered CONTEXT passages. Do not use outside knowledge, even if you know the answer.
2. After every claim, cite the passage number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "{REFUSAL}" Do not guess.
4. Every passage has a STATUS. If you use a passage whose status is "withdrawn" or "marked_withdrawn", say clearly that the source is withdrawn / possibly superseded, so the user must verify the current rule.
5. If passages from different documents give different values (for example an older and a newer version), mention both and say which document each value comes from.
6. Passages may come from different bank types (commercial banks vs small finance banks). Say which one your answer applies to.
7. Be concise. Numbers, limits and timelines must be copied exactly from the context."""


def build_context(results):
    blocks = []
    for n, r in enumerate(results, start=1):
        pages = f"p{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp{r['page_start']}-{r['page_end']}"
        where = " > ".join(x for x in (r["section"] or r["chapter"], f"Para {r['para']}" if r["para"] else "") if x)
        blocks.append(
            f"[{n}] SOURCE: {r['title']} | {where} | {pages} | STATUS: {r['status']}\n{r['text']}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question, results):
    return f"CONTEXT:\n{build_context(results)}\n\nQUESTION: {question}\n\nANSWER (with [n] citations):"


def _models():
    raw = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    return [m.strip() for m in raw.split(",") if m.strip()]


def ask_gemini(question, results, timeout_s=45):
    """Try each model in GEMINI_MODELS; a busy model (503/504/429) is skipped after `timeout_s`
    instead of making the user wait minutes. Returns (answer_text, model_that_answered)."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing. Put it in .env as GEMINI_API_KEY=... and save.")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1)
    prompt = build_user_prompt(question, results)

    last_error = None
    for model in _models():
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            if resp.text:
                return resp.text, model
            last_error = f"{model} returned an empty answer"
        except Exception as e:  # busy / quota / timeout -> try the next model
            last_error = f"{model}: {str(e)[:120]}"
            print(f"[{model} failed -> trying next model]  {str(e)[:80]}")
    raise RuntimeError(f"All Gemini models failed or are busy. Try again in a minute. Last error: {last_error}")


def answer(question, k=5, mode="hybrid", active_only=False, retriever=None, rerank=True,
           prefer_current=True, timings=None):
    """Defaults = best configuration from eval/history.json: hybrid + cross-encoder rerank + version-aware order."""
    t0 = time.time()
    retriever = retriever or Retriever()
    results = retriever.search(question, k=k, mode=mode, active_only=active_only,
                               rerank=rerank, prefer_current=prefer_current)
    t1 = time.time()
    if timings is not None:
        timings["search_s"] = t1 - t0
    if not results:
        return REFUSAL, []
    text, model = ask_gemini(question, results)
    if timings is not None:
        timings["gemini_s"] = time.time() - t1   # tells us WHERE the time goes: search or the LLM call
        timings["model"] = model
    return text, results


def print_sources(results):
    print("\nSources:")
    for n, r in enumerate(results, start=1):
        pages = f"p{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp{r['page_start']}-{r['page_end']}"
        where = r["section"] or r["chapter"]
        print(f"  [{n}] {r['title']} | {where[:60]} | {pages} | {r['status']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--active-only", action="store_true")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (faster)")
    ap.add_argument("--no-prefer-current", action="store_true", help="do not push withdrawn versions down")
    a = ap.parse_args()
    try:
        timings = {}
        text, results = answer(a.question, a.k, a.mode, a.active_only,
                               rerank=not a.no_rerank, prefer_current=not a.no_prefer_current,
                               timings=timings)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"Gemini call failed: {e}\n"
            "Check: (1) key is valid, (2) model names in GEMINI_MODELS (run src\\gemini_models.py), "
            "(3) free-tier quota / Google busy - wait a minute and retry."
        )
    print("\n" + text)
    print(f"\n[time: search {timings.get('search_s', 0):.1f} s | Gemini {timings.get('gemini_s', 0):.1f} s | model {timings.get('model', '-')}]")
    if REFUSAL.lower() not in text.lower():   # no sources for a refusal: they would look like evidence
        print_sources(results)

"""Helper for writing your own held-out questions.

Shows random chunks you have NOT already used in eval/questions.json, so you read a real
paragraph, write a question about it, and copy an exact phrase from it into eval/heldout.json.

Run:
  python src\\browse.py                       # 4 random chunks from any document
  python src\\browse.py --doc kyc_commercial_banks_2025 --n 3
  python src\\browse.py --n 6 --seed 7        # same seed = same chunks again
Documents: digital_lending_2025, kyc_commercial_banks_2025, kyc_small_finance_banks_2025,
           psl_directions_2025, psl_master_direction_2020
"""
import argparse
import json
import random
import re
from pathlib import Path

CHUNKS = Path("data/processed/chunks.jsonl")
QUESTIONS = Path("eval/questions.json")


def norm(t):
    return re.sub(r"\s+", " ", t.replace("’", "'")).lower()


def used_chunk_ids(chunks):
    """Chunks that are gold for one of my 40 questions: skip them, they are not 'held out'."""
    used = set()
    if QUESTIONS.exists():
        for q in json.loads(QUESTIONS.read_text(encoding="utf-8")):
            if not q.get("gold_doc"):
                continue
            for c in chunks:
                if c["doc_id"] == q["gold_doc"] and all(
                        norm(m) in norm(c["context"] + "\n" + c["text"]) for m in q["must_contain"]):
                    used.add(c["chunk_id"])
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    used = used_chunk_ids(chunks)
    pool = [c for c in chunks if c["chunk_id"] not in used and not c["is_annex"]
            and len(c["text"]) > 300 and (a.doc is None or c["doc_id"] == a.doc)]
    if not pool:
        raise SystemExit("No chunks match. Check the --doc name.")
    random.seed(a.seed)
    for c in random.sample(pool, min(a.n, len(pool))):
        print("=" * 90)
        print(f"gold_doc: {c['doc_id']}   |   {c['section'] or c['chapter']}   |   p{c['page_start']}")
        print("-" * 90)
        print(c["text"][:900])
    print("=" * 90)
    print("Pick a paragraph -> write a question in your own words -> copy a 4-8 word phrase EXACTLY from the text above.")


if __name__ == "__main__":
    main()

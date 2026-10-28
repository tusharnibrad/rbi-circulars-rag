"""Step 6: measure retrieval quality. This is what turns the project into evidence.

For every answerable question in eval/questions.json we retrieve the top-10 chunks and find the
rank of the first GOLD chunk = a chunk from the gold document whose text contains every
`must_contain` phrase. Metrics:
  Hit@k   share of questions with a gold chunk in the top k      (did we find the evidence?)
  MRR     mean of 1/rank of the first gold chunk                  (how high did we rank it?)
  DocHit@5 gold DOCUMENT present in top 5                         (right document, maybe wrong chunk)

Run from the project root:
  python eval\\run_eval.py --label baseline
  python eval\\run_eval.py --modes hybrid --no-annex --label hybrid_no_annex
  python eval\\run_eval.py --modes hybrid --rerank --label hybrid_rerank
  python eval\\run_eval.py --modes hybrid --rerank --prefer-current --label rerank_current
  python eval\\run_eval.py --modes hybrid --rerank --prefer-current --questions eval\\heldout.json --label heldout
Results are appended to eval/history.json so you can show "baseline -> improved" in the README.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieve import Retriever  # noqa: E402

DEFAULT_QUESTIONS = "eval/questions.json"
HISTORY = Path("eval/history.json")
K_MAX = 10


def norm(t):
    return re.sub(r"\s+", " ", t.replace("’", "'")).lower()


def is_gold(chunk, q):
    if chunk["doc_id"] != q["gold_doc"]:
        return False
    hay = norm(chunk["context"] + "\n" + chunk["text"])
    return all(norm(m) in hay for m in q["must_contain"])


def first_rank(results, pred):
    for i, r in enumerate(results, start=1):
        if pred(r):
            return i
    return None


def evaluate(retriever, questions, mode, no_annex, rerank=False, prefer_current=False):
    rows = []
    for q in questions:
        res = retriever.search(q["question"], k=K_MAX, mode=mode, exclude_annex=no_annex, rerank=rerank,
                           prefer_current=prefer_current)
        rows.append({
            "id": q["id"], "category": q["category"],
            "gold_rank": first_rank(res, lambda r: is_gold(r, q)),
            "doc_rank": first_rank(res, lambda r: r["doc_id"] == q["gold_doc"]),
            "top1": f"{res[0]['doc_id']} | {(res[0]['section'] or res[0]['chapter'])[:40]}" if res else "-",
        })
    return rows


def summarize(rows):
    n = len(rows)
    hit = lambda k: sum(1 for r in rows if r["gold_rank"] and r["gold_rank"] <= k) / n
    return {
        "n": n,
        "hit@1": round(hit(1), 3), "hit@3": round(hit(3), 3), "hit@5": round(hit(5), 3),
        "mrr": round(sum(1 / r["gold_rank"] for r in rows if r["gold_rank"]) / n, 3),
        "dochit@5": round(sum(1 for r in rows if r["doc_rank"] and r["doc_rank"] <= 5) / n, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--label", default="run")
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS, help="e.g. eval/heldout.json (your own questions)")
    ap.add_argument("--no-annex", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--prefer-current", action="store_true")
    ap.add_argument("--show-failures", type=int, default=6)
    a = ap.parse_args()

    all_q = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    answerable = [q for q in all_q if q["gold_doc"]]
    print(f"{len(answerable)} answerable questions ({len(all_q) - len(answerable)} unanswerable are tested in answer.py)")

    retriever = Retriever()
    # A typo in gold_doc / must_contain would silently count as a miss. Catch it up front.
    missing = [q["id"] for q in answerable if not any(is_gold(c, q) for c in retriever.chunks)]
    if missing:
        print(f"WARNING: no chunk matches gold_doc + must_contain for {missing}\n"
              "         fix those questions first, otherwise they always count as misses.\n")
    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else []

    print(f"\n{'mode':20} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'DocHit@5':>9}")
    per_mode = {}
    for mode in a.modes:
        name = mode + ("+rerank" if a.rerank else "") + ("+current" if a.prefer_current else "")
        rows = evaluate(retriever, answerable, mode, a.no_annex, a.rerank, a.prefer_current)
        s = summarize(rows)
        per_mode[name] = rows
        print(f"{name:20} {s['hit@1']:7.1%} {s['hit@3']:7.1%} {s['hit@5']:7.1%} {s['mrr']:7.3f} {s['dochit@5']:9.1%}")
        history.append({"label": a.label, "questions": a.questions, "mode": name, "no_annex": a.no_annex, **s})

    best = list(per_mode)[-1]
    print(f"\nBy category ({best}):  Hit@3")
    by_cat = defaultdict(list)
    for r in per_mode[best]:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        h3 = sum(1 for r in rs if r["gold_rank"] and r["gold_rank"] <= 3) / len(rs)
        print(f"  {cat:12} n={len(rs):2}  Hit@3={h3:.0%}")

    fails = [r for r in per_mode[best] if not r["gold_rank"] or r["gold_rank"] > 3]
    print(f"\nFailures in top-3 ({best}): {len(fails)}  <- read these, this is your error analysis")
    for r in fails[: a.show_failures]:
        print(f"  {r['id']:20} gold_rank={r['gold_rank']}  doc_rank={r['doc_rank']}  top1 -> {r['top1']}")

    HISTORY.write_text(json.dumps(history, indent=2), encoding="utf-8")
    Path("eval/last_run.json").write_text(json.dumps(per_mode, indent=2), encoding="utf-8")
    print(f"\nSaved -> {HISTORY} and eval/last_run.json")


if __name__ == "__main__":
    main()

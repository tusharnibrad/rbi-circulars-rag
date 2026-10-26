"""Step 4 of the pipeline: question -> top-k relevant chunks.

Three modes so you can MEASURE each one in the evaluation step:
  dense   - embeddings only
  bm25    - keywords only
  hybrid  - both, merged with Reciprocal Rank Fusion (RRF)

Try it:
  python src\\retrieve.py "What is the cooling-off period for a digital loan?"
  python src\\retrieve.py "KYC for small finance banks V-CIP" --mode bm25 --k 3
  python src\\retrieve.py "priority sector target for agriculture" --active-only
  python src\\retrieve.py "cooling-off period" --rerank        (re-orders the top 20 with a cross-encoder)
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from index import CHUNKS_FILE, INDEX_DIR, MODEL_NAME, load_chunks, tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60          # standard RRF constant; larger = flatter weighting of top ranks
POOL = 50           # candidates taken from each retriever before fusing / filtering
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small (~90 MB), fast on CPU
RERANK_POOL = 20    # how many fused candidates the cross-encoder re-reads
# Words that mean "I am asking about an older rule" -> do not push old documents down
HISTORY_RE = re.compile(r"\b(old|older|previous|previously|earlier|former|formerly|superseded|withdrawn|was|were)\b", re.I)


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.faiss = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with (INDEX_DIR / "bm25_tokens.pkl").open("rb") as f:
            self.bm25 = BM25Okapi(pickle.load(f))
        self._model = None
        self._reranker = None
        # A document is "superseded" if it is not active AND its topic has an active (newer) document.
        # (Digital Lending is marked withdrawn but has no newer document here, so it is NOT superseded.)
        active_topics = {c["topic"] for c in self.chunks if c["status"] == "active"}
        self.superseded = {c["doc_id"] for c in self.chunks
                           if c["status"] != "active" and c["topic"] in active_topics}

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        return self._reranker

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    # ---- single retrievers: each returns a ranked list of (chunk_index, score) ----
    def dense(self, query, pool=POOL):
        q = self.model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        scores, ids = self.faiss.search(q, pool)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query, pool=POOL):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:pool]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def rrf(*rankings):
        """Reciprocal Rank Fusion: score = sum over retrievers of 1/(RRF_K + rank)."""
        fused = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ---- public API ----
    def search(self, query, k=5, mode="hybrid", active_only=False, exclude_annex=False,
               rerank=False, pool=RERANK_POOL, prefer_current=False):
        """rerank=True: take the top `pool` candidates, let a cross-encoder read (query, chunk)
        together and re-order them. Slower, but fixes ordering mistakes of the first stage."""
        if mode == "dense":
            ranked = self.dense(query)
        elif mode == "bm25":
            ranked = self.sparse(query)
        else:
            ranked = self.rrf(self.dense(query), self.sparse(query))

        want = max(pool, k) if (rerank or prefer_current) else k
        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            if active_only and c["status"] != "active":
                continue
            if exclude_annex and c["is_annex"]:
                continue
            results.append({**c, "score": score})
            if len(results) == want:
                break
        if rerank and results:
            pairs = [(query, f"{r['context']}\n{r['text']}") for r in results]
            for r, sc in zip(results, self.reranker.predict(pairs)):
                r["fusion_score"], r["score"] = r["score"], float(sc)
            results.sort(key=lambda r: r["score"], reverse=True)
        if prefer_current:
            results = self._prefer_current(query, results)
        return results[:k]

    def _prefer_current(self, query, results):
        """Version-aware ordering: chunks of superseded documents go behind current ones, unless the
        question names that document's year ("... 2020 ...") or asks about the past ("was", "earlier").
        Stable: order inside each group is unchanged."""
        if HISTORY_RE.search(query):
            return results
        def is_old(r):
            return r["doc_id"] in self.superseded and str(r["year"]) not in query
        return [r for r in results if not is_old(r)] + [r for r in results if is_old(r)]


def show(results):
    for rank, r in enumerate(results, start=1):
        where = r["section"] or r["chapter"]
        para = f" Para {r['para']}" if r["para"] and not r["section"].startswith(r["para"] + ".") else ""
        pages = f"p{r['page_start']}" if r["page_start"] == r["page_end"] else f"p{r['page_start']}-{r['page_end']}"
        print(f"\n#{rank}  score={r['score']:.4f}  [{r['status']}]  {r['doc_id']}  {pages}")
        print(f"    {where[:70]}{para}")
        print("    " + r["text"][:300].replace("\n", " ") + ("..." if len(r["text"]) > 300 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--active-only", action="store_true")
    ap.add_argument("--no-annex", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--prefer-current", action="store_true")
    a = ap.parse_args()
    show(Retriever().search(a.query, a.k, a.mode, a.active_only, a.no_annex,
                            rerank=a.rerank, prefer_current=a.prefer_current))

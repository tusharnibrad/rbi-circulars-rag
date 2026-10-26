"""Step 3 of the pipeline: chunks -> search indexes.

Builds two indexes over the same chunks:
  - dense (meaning):   BGE embeddings + FAISS      -> finds paraphrases ("cooling off" ~ "exit the loan")
  - sparse (keywords): BM25                        -> finds exact terms ("V-CIP", "DLG", "para 27")
Hybrid retrieval (retrieve.py) combines both, because legal text needs both.

Run:  python src\\index.py        (first run downloads the ~130 MB model, then takes 1-3 min on CPU)
Output: data/index/faiss.index, data/index/bm25_tokens.pkl, data/index/index_info.json
"""
import json
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")   # keeps "v-cip", "co-lending", "e-pan"


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def embed_text(chunk: dict) -> str:
    """What we embed: the context line (document | chapter | section) + the chunk text."""
    return f"{chunk['context']}\n{chunk['text']}"


def load_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    if not CHUNKS_FILE.exists():
        print(f"{CHUNKS_FILE} not found. Run: python src\\chunk.py")
        return
    chunks = load_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(chunks)} chunks")

    # 1) dense index
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [embed_text(c) for c in chunks],
        batch_size=32,
        normalize_embeddings=True,   # unit vectors -> inner product == cosine similarity
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])   # exact search; fine for thousands of chunks
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"Dense index: {index.ntotal} vectors, dim={vectors.shape[1]}  ({time.time()-t0:.0f}s)")

    # 2) keyword index (BM25 is rebuilt from these tokens at load time; it is instant)
    tokens = [tokenize(embed_text(c)) for c in chunks]
    with (INDEX_DIR / "bm25_tokens.pkl").open("wb") as f:
        pickle.dump(tokens, f)
    print(f"BM25 tokens saved for {len(tokens)} chunks")

    info = {"model": MODEL_NAME, "n_chunks": len(chunks), "dim": int(vectors.shape[1])}
    (INDEX_DIR / "index_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done -> {INDEX_DIR}")


if __name__ == "__main__":
    main()

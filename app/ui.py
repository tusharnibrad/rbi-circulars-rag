"""Streamlit app: ask questions about RBI circulars, get a cited answer.

Run locally:   streamlit run app\\ui.py
Secrets:       GEMINI_API_KEY (in .env locally; as a "secret" on Streamlit Cloud / Hugging Face)
"""
import sys
import time
from collections import OrderedDict
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from answer import REFUSAL, answer  # noqa: E402
from retrieve import Retriever  # noqa: E402

MAX_QUESTIONS_PER_SESSION = 15   # protects the public demo's API quota
BADGE = {"active": "🟢 active", "withdrawn": "🔴 withdrawn", "marked_withdrawn": "🟠 marked withdrawn"}
EXAMPLES = [
    "What is the cooling-off period for a digital loan?",
    "How often must a commercial bank update KYC for high-risk customers?",
    "What is the education loan limit under priority sector lending?",
    "What was the education loan limit under the 2020 Master Directions?",
]

st.set_page_config(page_title="RBI Circulars Q&A", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading search models (first run only, ~30 s)...")
def get_retriever():
    r = Retriever()
    r.search("warm up", k=1, rerank=True)   # download/load both models once, not on the first user question
    return r


def pages_label(r):
    return f"p{r['page_start']}" if r["page_start"] == r["page_end"] else f"pp{r['page_start']}-{r['page_end']}"


retriever = get_retriever()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Passages given to the LLM", 3, 8, 5)
    use_rerank = st.toggle("Cross-encoder reranking", value=True, help="More accurate, slightly slower")
    prefer_current = st.toggle("Prefer current versions", value=True, help="Pushes withdrawn documents down")
    st.header("Documents indexed")
    docs = OrderedDict()
    for c in retriever.chunks:
        docs.setdefault(c["doc_id"], (c["title"], c["status"]))
    for title, status in docs.values():
        st.caption(f"{BADGE.get(status, status)}  \n{title}")
    st.divider()
    st.caption("Built with PyMuPDF, BGE embeddings, FAISS, BM25, a cross-encoder reranker and Gemini. "
               "Not legal advice: always verify on rbi.org.in.")

# ---------------- main ----------------
st.title("🏦 RBI Circulars Q&A")
st.write("Ask about KYC, Digital Lending or Priority Sector Lending. Every answer cites its source passages, "
         "and withdrawn documents are flagged.")

st.session_state.setdefault("question", "")
st.session_state.setdefault("count", 0)

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, use_container_width=True, on_click=lambda q=ex: st.session_state.update(question=q))

question = st.text_input("Your question", key="question", placeholder="e.g. What is the DLG cap in digital lending?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Demo limit reached for this session. Refresh the page to continue.")
        st.stop()
    st.session_state.count += 1
    t0 = time.time()
    timings = {}
    try:
        with st.spinner("Searching the documents and writing the answer..."):
            text, results = answer(question.strip(), k=k, retriever=retriever,
                                   rerank=use_rerank, prefer_current=prefer_current, timings=timings)
    except Exception as e:   # bad key, wrong model name, quota
        st.error(f"Could not get an answer: {e}")
        st.stop()
    elapsed = time.time() - t0

    st.subheader("Answer")
    st.markdown(text)
    st.caption(f"{elapsed:.1f} s total (search {timings.get('search_s', 0):.1f} s · Gemini {timings.get('gemini_s', 0):.1f} s · {timings.get('model', '-')})"
               f" · {len(results)} passages · rerank {'on' if use_rerank else 'off'}")

    if results and REFUSAL.lower() not in text.lower():
        st.subheader("Sources")
        for n, r in enumerate(results, start=1):
            where = r["section"] or r["chapter"]
            with st.expander(f"[{n}] {r['title']} · {where[:60]} · {pages_label(r)} · {BADGE.get(r['status'], r['status'])}"):
                if r["status"] != "active":
                    st.warning(r["status_note"] or "This document is withdrawn or possibly superseded.")
                st.text(r["text"])

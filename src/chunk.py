"""Step 2 of the pipeline: page text -> chunks for retrieval.

Why not fixed-size chunking?
RBI documents are built from chapters, numbered paragraphs and clauses like
(i), (ii), (a). A fixed 500-character window cuts clauses in half and mixes two
topics in one chunk. Here we:
  1. detect chapter / section headings,
  2. group lines into clauses (one clause = one paragraph),
  3. pack whole clauses into chunks of at most MAX_CHARS, never crossing a section,
  4. repeat the last clause of a chunk at the start of the next one (overlap),
  5. keep a `context` string (document | chapter | section) that is added to the
     text when embedding, so "5.1 targets" is never confused across documents.

Run:  python src\\chunk.py
Input:  data/processed/pages.jsonl   (from ingest.py)
Output: data/processed/chunks.jsonl
"""
import json
import re
from collections import defaultdict
from pathlib import Path

PAGES_FILE = Path("data/processed/pages.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")

MAX_CHARS = 1500      # ~350 tokens: big enough for a full clause, small enough to stay focused
MIN_CHARS = 80        # chunks shorter than this are merged or dropped
OVERLAP_MAX = 400     # carry the last clause into the next chunk only if it is short

CHAPTER_RE = re.compile(r"^(CHAPTER|Chapter|Chatper)\s*[–-]?\s*[IVXLC\d]+\b")
NUM_HEAD_RE = re.compile(r"^(\d{1,2})\.\s*([A-Z].{2,90})$")      # "5. Targets for Priority Sector"
LETTER_HEAD_RE = re.compile(r"^([A-Z])\.\s+([A-Z].{2,90})$")      # "A. CDD Procedure"
PARA_NO_RE = re.compile(r"^(\d{1,2})\.\s")
ANNEX_RE = re.compile(r"^(ANNEX|Annex|APPENDIX|Appendix)\b")
MODAL_RE = re.compile(r"\b(shall|should|must|may|will|is|are|has|have|being)\b", re.IGNORECASE)
CLAUSE_START_RE = re.compile(
    r"^(\(?[ivxlc]{1,5}[\).]|\([a-z]\)|[a-z]\.|\d{1,2}\.\d{1,2}(\.\d{1,2})?\.?|\d{1,2}\.|[A-Z]\.)\s"
)


def is_heading(line: str):
    """Return the heading text if `line` looks like a section heading, else None."""
    m = NUM_HEAD_RE.match(line) or LETTER_HEAD_RE.match(line)
    if not m:
        return None
    body = m.group(2).strip()
    # A real heading is short and has no verbs. "27. The bank shall adhere to ..." is a
    # clause whose first line happens to be short, not a heading.
    if body[-1] in ",;" or len(body.split()) > 12 or MODAL_RE.search(body):
        return None
    return f"{m.group(1)}. {body.rstrip('.:')}"


def split_long(text: str, max_chars: int):
    """Split an oversized clause on sentence boundaries."""
    if len(text) <= max_chars * 1.5:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;])\s+", text):
        if cur and len(cur) + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    # no sentence breaks (tables, district lists): fall back to a word-boundary split
    final = []
    for part in parts:
        while len(part) > max_chars * 1.5:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            final.append(part[:cut])
            part = part[cut:].strip()
        final.append(part)
    return final


def join_lines(lines):
    """Join wrapped lines; a line ending in '-' keeps its compound hyphen (Non-Banking)."""
    text = ""
    for ln in lines:
        text = ln if not text else (text + ln if text.endswith("-") else f"{text} {ln}")
    return text


def doc_to_clauses(pages):
    """pages (one document) -> list of clauses with chapter/section/page info."""
    clauses = []
    chapter, section = "", ""
    cur_lines, cur_page = [], None
    para = ""             # top-level paragraph number inside a section, e.g. "27"
    expect_title = None   # None | "chapter" | "annex"

    # Running headers/footers repeat on many pages; they are noise inside clauses.
    seen = defaultdict(int)
    for rec in pages:
        for ln in set(rec["text"].splitlines()):
            seen[ln] += 1
    repeat_min = max(5, len(pages) // 5)
    running = {ln for ln, n in seen.items() if n >= repeat_min and len(ln) < 140}

    para_of_clause = ""

    def flush():
        nonlocal cur_lines, cur_page
        if cur_lines:
            text = join_lines(cur_lines)
            for piece in split_long(text, MAX_CHARS):
                clauses.append({"text": piece, "page": cur_page, "chapter": chapter,
                                "section": section, "para": para_of_clause})
        cur_lines, cur_page = [], None

    for rec in pages:
        for line in rec["text"].splitlines():
            if line in running:
                continue
            if expect_title:
                mode, expect_title = expect_title, None
                title_ok = line.isupper() if mode == "chapter" else not CLAUSE_START_RE.match(line)
                if title_ok and len(line) < 120:   # "CATEGORIES AND TARGETS UNDER ..."
                    chapter = f"{chapter} {line.title() if mode == 'chapter' else line}"
                    continue
            if CHAPTER_RE.match(line) or ANNEX_RE.match(line):
                flush()
                chapter, section, para = line, "", ""
                if len(line) < 20:
                    expect_title = "annex" if ANNEX_RE.match(line) else "chapter"
                continue
            head = is_heading(line)
            if head:
                flush()
                section, para = head, ""
                continue
            if CLAUSE_START_RE.match(line) and cur_lines:
                flush()
            if not cur_lines:
                cur_page = rec["page"]
                m = PARA_NO_RE.match(line)
                if m and not ANNEX_RE.match(chapter):   # annex lists are items, not paragraphs
                    para = m.group(1)
                para_of_clause = para
            cur_lines.append(line)
    flush()
    return clauses


def pack(clauses):
    """Group clauses of the same (chapter, section) into chunks of <= MAX_CHARS."""
    groups, key, bucket = [], None, []
    for c in clauses:
        k = (c["chapter"], c["section"], c["para"])
        if k != key and bucket:
            tiny = sum(len(x["text"]) for x in bucket) < MIN_CHARS
            prev = groups[-1][-1] if groups else None
            if tiny and prev and (prev["chapter"], prev["section"]) == (bucket[0]["chapter"], bucket[0]["section"]):
                groups[-1].extend(bucket)     # don't lose short paragraphs
            else:
                groups.append(bucket)
            bucket = []
        key = k
        bucket.append(c)
    if bucket:
        groups.append(bucket)

    chunks = []
    for group in groups:
        cur, cur_len = [], 0
        for c in group:
            n = len(c["text"])
            if cur and cur_len + n > MAX_CHARS:
                chunks.append(cur)
                last = cur[-1]
                cur = [last] if len(last["text"]) <= OVERLAP_MAX else []
                cur_len = sum(len(x["text"]) for x in cur)
            cur.append(c)
            cur_len += n
        if cur:
            chunks.append(cur)
    return chunks


def build_chunks(doc_pages):
    meta = doc_pages[0]
    out = []
    for n, group in enumerate(pack(doc_to_clauses(doc_pages)), start=1):
        text = "\n".join(c["text"] for c in group)
        if len(text) < MIN_CHARS:
            continue
        chapter, section, para = group[0]["chapter"], group[0]["section"], group[0]["para"]
        para_label = f"Para {para}" if para and not section.startswith(f"{para}.") else ""
        context = " | ".join(x for x in (meta["title"], chapter, section, para_label) if x)
        out.append({
            "chunk_id": f"{meta['doc_id']}_{n:04d}",
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "topic": meta["topic"],
            "year": meta["year"],
            "status": meta["status"],
            "status_note": meta["status_note"],
            "chapter": chapter,
            "section": section,
            "para": para,
            "is_annex": bool(ANNEX_RE.match(chapter)),
            "page_start": min(c["page"] for c in group),
            "page_end": max(c["page"] for c in group),
            "context": context,   # embed: context + "\n" + text
            "text": text,
        })
    return out


def main():
    if not PAGES_FILE.exists():
        print(f"{PAGES_FILE} not found. Run: python src\\ingest.py")
        return

    by_doc = defaultdict(list)
    for line in PAGES_FILE.open(encoding="utf-8"):
        rec = json.loads(line)
        by_doc[rec["doc_id"]].append(rec)

    all_chunks = []
    for doc_id, pages in by_doc.items():
        chunks = build_chunks(pages)
        lens = [len(c["text"]) for c in chunks]
        print(f"{doc_id:34} chunks={len(chunks):4}  avg_chars={sum(lens)//max(len(lens),1):5}  max={max(lens)}")
        all_chunks.extend(chunks)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks)} chunks -> {OUT_FILE}")

    print("\n--- 3 sample chunks (check: does each one read as a complete, single topic?) ---")
    step = max(len(all_chunks) // 3, 1)
    for c in all_chunks[step // 2::step][:3]:
        print(f"\n[{c['chunk_id']}] pages {c['page_start']}-{c['page_end']}")
        print(f"context: {c['context']}")
        print(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))


if __name__ == "__main__":
    main()

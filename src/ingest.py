"""Step 1 of the pipeline: PDF -> clean page text.

- Text PDFs: read the text layer with PyMuPDF.
- Scanned pages (almost no text layer): OpenCV preprocessing + Tesseract OCR.
- Removes noise: "Withdrawn" watermark, Hindi header, bare page numbers, TOC dot-lines.
- Attaches document metadata (title, status) to every page.

Run:  python src\\ingest.py
Output: data/processed/pages.jsonl  (one JSON object per page)
"""
import json
import re
from pathlib import Path

import pymupdf  # PyMuPDF (new import name, replaces `import fitz`)

RAW_DIR = Path("data/raw_pdfs")
OUT_FILE = Path("data/processed/pages.jsonl")

MIN_CHARS = 50   # a page with fewer characters than this is treated as scanned
OCR_DPI = 300    # 300 dpi is the sweet spot for Tesseract

# Metadata per document. `status` matters: old and new versions of the same
# regulation conflict, so retrieval must know which one is current.
DOC_META = {
    "digital_lending_2025": {
        "title": "RBI Digital Lending Directions, 2025",
        "topic": "digital_lending",
        "year": 2025,
        "status": "marked_withdrawn",
        "status_note": "PDF carries a 'Withdrawn' watermark. Check on rbi.org.in what replaced it.",
    },
    "psl_directions_2025": {
        "title": "RBI Priority Sector Lending (Targets and Classification) Directions, 2025",
        "topic": "priority_sector_lending",
        "year": 2025,
        "status": "active",
        "status_note": "Effective April 1, 2025; supersedes the 2020 Master Directions.",
    },
    "psl_master_direction_2020": {
        "title": "RBI Priority Sector Lending Master Directions, 2020",
        "topic": "priority_sector_lending",
        "year": 2020,
        "status": "withdrawn",
        "status_note": "Withdrawn w.e.f. April 1, 2025; replaced by the 2025 Directions.",
    },
    "kyc_commercial_banks_2025": {
        "title": "RBI KYC Directions, 2025 (Commercial Banks)",
        "topic": "kyc",
        "year": 2025,
        "status": "active",
        "status_note": "Applies to commercial banks.",
    },
    "kyc_small_finance_banks_2025": {
        "title": "RBI KYC Directions, 2025 (Small Finance Banks)",
        "topic": "kyc",
        "year": 2025,
        "status": "active",
        "status_note": "Applies to small finance banks.",
    },
}

WATERMARK_RE = re.compile(r"^\s*Withdrawn(\s+w\.?\s?e\.?\s?f\.?.*)?\s*$", re.IGNORECASE)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TOC_DOTS_RE = re.compile(r"\.{5,}")
BARE_NUMBER_RE = re.compile(r"^\d{1,3}\.?$")   # "12" or "12." alone on a line
TOC_PAGE_MIN_LEADERS = 5  # page with this many dotted-leader lines is a table of contents


def clean_text(text: str) -> str:
    """Drop lines that are noise, keep everything else untouched."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if WATERMARK_RE.match(s):
            continue
        if DEVANAGARI_RE.search(s) or "\ufffd" in s:
            continue
        if BARE_NUMBER_RE.match(s):
            continue
        if TOC_DOTS_RE.search(s):
            continue
        kept.append(s)
    return "\n".join(kept)


def ocr_page(page) -> str:
    """Scanned page -> text. Imports are lazy so text-only runs never need Tesseract."""
    import cv2
    import numpy as np
    import pytesseract

    # Windows: Tesseract is often installed but not on PATH
    default_win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_win.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_win)

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = img[:, :, 0] if pix.n == 1 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Adaptive threshold handles uneven lighting / shadows on scans
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return pytesseract.image_to_string(binary, lang="eng", config="--psm 6")


def extract_pdf(pdf_path: Path):
    doc_id = pdf_path.stem
    meta = DOC_META.get(
        doc_id,
        {"title": doc_id.replace("_", " "), "topic": "unknown", "year": None,
         "status": "unknown", "status_note": ""},
    )
    records = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text()
            method = "text"
            if len(raw.strip()) < MIN_CHARS:
                try:
                    raw = ocr_page(page)
                    method = "ocr"
                except Exception as e:  # Tesseract missing etc.
                    print(f"  ! OCR failed on {doc_id} p{i}: {e}")
                    method = "ocr_failed"
            # Table-of-contents pages only add noise to retrieval
            if sum(bool(TOC_DOTS_RE.search(l)) for l in raw.splitlines()) >= TOC_PAGE_MIN_LEADERS:
                continue
            text = clean_text(raw)
            if not text:
                continue
            records.append({"doc_id": doc_id, **meta, "page": i, "method": method, "text": text})
    return records


def main():
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in '{RAW_DIR}'. Add RBI PDFs first.")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_pages = total_ocr = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            recs = extract_pdf(pdf)
            ocr_n = sum(r["method"].startswith("ocr") for r in recs)
            chars = sum(len(r["text"]) for r in recs)
            print(f"{pdf.name:42} pages={len(recs):4}  ocr_pages={ocr_n:3}  chars={chars:8}")
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_pages += len(recs)
            total_ocr += ocr_n

    print(f"\nSaved {total_pages} pages ({total_ocr} via OCR) -> {OUT_FILE}")


if __name__ == "__main__":
    main()

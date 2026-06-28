"""
Extract structured chunks from URA planning PDFs.

Reads every PDF in data/raw/, extracts tables (with forward-filled parameter
column and dropped empty columns/rows) AND prose paragraphs from every page,
then writes one JSONL chunk per line to data/processed/chunks.jsonl.

Each chunk has:
- chunk_id:    unique identifier (filename + page + index)
- source_file: PDF filename
- page:        page number (0-indexed)
- chunk_type:  "table_row" or "prose"
- parameter:   the row's parameter name (table_row only)
- text:        human-readable string used for embedding
- raw_cells:   the original cell list (table_row only, for debugging)

Design notes:
- On every page we extract BOTH tables and prose. To avoid duplicating table
  content as garbled prose chunks (page.extract_text() returns all text on the
  page, including table cells), we collect table cell words and skip any
  paragraph whose words substantially overlap with them.
- Header rows ("Parameter | Guideline") aren't auto-detected; row 0 of each
  cleaned table is skipped under the assumption it's a header. This is true
  for URA's Summary handbooks but may not generalize.
- Multi-page tables: each page's portion is treated independently. Cross-page
  forward-fill is not attempted.
"""

import json
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")
MIN_PROSE_CHARS = 50
TABLE_OVERLAP_THRESHOLD = 0.5  # drop paragraph if >=50% of its words appear in table cells


def clean_table(table):
    """Drop empty columns and rows; forward-fill the parameter column."""
    if not table or not table[0]:
        return []

    n_cols = max(len(row) for row in table)
    keep_cols = []
    for c in range(n_cols):
        col_vals = [row[c] if c < len(row) else None for row in table]
        if any(v not in (None, "") for v in col_vals):
            keep_cols.append(c)

    filtered = [
        [row[c] if c < len(row) else None for c in keep_cols] for row in table
    ]

    # forward filling
    last_param = None
    for row in filtered:
        if row and row[0] not in (None, ""):
            last_param = row[0]
        elif row:
            row[0] = last_param

    filtered = [r for r in filtered if any(v not in (None, "") for v in r)]
    return filtered


def row_to_text(row):
    cells = [str(c).replace("\n", " ").strip() for c in row if c not in (None, "")]
    return " | ".join(cells)


def collect_table_words(tables):
    """Lowercased set of words found in any table cell (length > 3)."""
    words = set()
    for table in tables:
        for row in table:
            for cell in row:
                if cell:
                    for w in str(cell).lower().split():
                        if len(w) > 3:
                            words.add(w)
    return words


def split_prose(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def is_table_residue(paragraph, table_words):
    if not table_words:
        return False
    words = [w for w in paragraph.lower().split() if len(w) > 3]
    if not words:
        return True
    overlap = sum(1 for w in words if w in table_words) / len(words)
    return overlap >= TABLE_OVERLAP_THRESHOLD


def process_pdf(pdf_path):
    pdf_name = pdf_path.stem

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            # Table chunks
            for t_idx, table in enumerate(tables):
                cleaned = clean_table(table)
                for r_idx, row in enumerate(cleaned[1:], start=1):
                    text = row_to_text(row)
                    if not text:
                        continue
                    yield {
                        "chunk_id": f"{pdf_name}_p{page_idx}_t{t_idx}_r{r_idx}",
                        "source_file": pdf_path.name,
                        "page": page_idx,
                        "chunk_type": "table_row",
                        "parameter": str(row[0]) if row[0] else None,
                        "text": text,
                        "raw_cells": [
                            str(c) if c is not None else "" for c in row
                        ],
                    }

            # Prose chunks (filtered against table content)
            table_words = collect_table_words(tables)
            text = page.extract_text() or ""
            for p_idx, paragraph in enumerate(split_prose(text)):
                if len(paragraph) < MIN_PROSE_CHARS:
                    continue
                if is_table_residue(paragraph, table_words):
                    continue
                yield {
                    "chunk_id": f"{pdf_name}_p{page_idx}_para{p_idx}",
                    "source_file": pdf_path.name,
                    "page": page_idx,
                    "chunk_type": "prose",
                    "text": paragraph,
                }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {RAW_DIR}")

    # pdfs = [RAW_DIR / "Summary_GFA.pdf"]
    # print(f"Found {len(pdfs)} PDFs in {RAW_DIR}")

    # pdfs = [
    # RAW_DIR / "MP25WrittenStatement.pdf",
    # RAW_DIR / "dc25-11_DTC.pdf",  # adjust if filename differs
    # ]

    n_chunks = n_table = n_prose = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for pdf_path in pdfs:
            count_before = n_chunks
            for chunk in process_pdf(pdf_path):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1
                if chunk["chunk_type"] == "table_row":
                    n_table += 1
                else:
                    n_prose += 1
            print(f"  {pdf_path.name}: {n_chunks - count_before} chunks")

    print(f"\nTotal: {n_chunks} chunks  ({n_table} table_row, {n_prose} prose)")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
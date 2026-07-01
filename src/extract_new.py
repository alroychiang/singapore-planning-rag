"""
Extract structured chunks from URA planning PDFs.

v3: header-aware extraction with checkbox table handling.

Changes from v2:
- Detect multi-row table headers (e.g. "Master / Plan / Control*" wrapped over
  6 rows). Reconstruct each column's full header by joining its non-empty cells.
- Detect "checkbox-style" tables: any data cell containing '✓'. For these,
  emit chunks like "Void Deck: Included as GFA" instead of "Void Deck | ✓".
- Carry headers forward across tables within the same document. Multi-page
  tables often lose the header on subsequent pages; using the most recent
  reconstructed headers from earlier in the doc keeps chunks meaningful.

Two table styles are handled:
  1. Parameter-value style (e.g. Summary-B1): pipe-joined cells, unchanged
     from v2. Example: "Road Buffer | Category 1 – Expressway | 15m"
  2. Checkbox style (e.g. Summary_GFA): "{item}: {categories with check}".
     Example: "Void Deck: Included as GFA"

Known limitations:
- Header reconstruction depends on the first table having visible header rows.
  If a PDF's very first table on page 0 lacks them, all chunks in that document
  will use placeholder headers (col_0, col_1, ...).
- Prose paragraphs with high word-overlap with table content are still dropped
  (see is_table_residue). Bbox-based prose extraction is the principled fix
  but deferred.
"""

import json
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")
MIN_PROSE_CHARS = 50
TABLE_OVERLAP_THRESHOLD = 0.5

# First-cell values that indicate "this row is still a header"
# this is "hardcoded". Manually/ visually identified first word in all tables in all documents' key word to identify header row
HEADER_FIRST_CELL = {"Items", "Item", "Parameter", "Parameters", "S/No"}


# ---------- Table function helpers ----------

def drop_empty_columns(table):
    """Drop columns where every cell is None or empty."""
    if not table:
        return []
    n_cols = max((len(row) for row in table), default=0)
    keep = [
        c for c in range(n_cols)
        if any(row[c] not in (None, "") for row in table if c < len(row))
    ]
    return [[row[c] if c < len(row) else None for c in keep] for row in table]


def split_header_data(table):
    """Split table rows into header rows and data rows.

    A row is a header if its first cell is empty/None OR matches a known
    header word ('Items', 'Parameter', ...). The first row that doesn't
    match either, marks the start of data.
    """
    headers, data = [], []
    in_header = True
    for row in table:
        first = row[0] if row else None
        first_str = str(first).strip() if first not in (None, "") else None
        if in_header and (first_str is None or first_str in HEADER_FIRST_CELL):
            headers.append(row)
        else:
            in_header = False
            data.append(row)
    return headers, data


def reconstruct_headers(header_rows, n_cols):
    """For each column, join non-empty cells across header rows (deduped).

    Handles multi-row wrapped headers. Example for GFA col 3:
      Row 0 col 3: 'Included as'
      Row 1 col 3: 'GFA'
      -> reconstructed header for col 3: 'Included as GFA'
    """
    headers = []
    for c in range(n_cols):
        parts = []
        for row in header_rows:
            v = row[c] if c < len(row) else None
            if v not in (None, ""):
                s = str(v).strip()
                if not parts or parts[-1] != s:
                    parts.append(s)
        headers.append(" ".join(parts) if parts else f"col_{c}")
    return headers


def is_checkbox_table(data_rows):
    """True if any data cell is the checkmark character."""
    return any(cell == "✓" for row in data_rows for cell in row)


def forward_fill_col0(data_rows):
    """In place: rows with empty col 0 inherit value from previous row."""
    last = None
    for row in data_rows:
        if row and row[0] not in (None, ""):
            last = row[0]
        elif row:
            row[0] = last


# ---------- Chunk emitters ----------

def emit_checkbox_chunks(data_rows, headers, pdf_name, pdf_filename, page_idx, t_idx):
    """For checkbox tables: '{item}: {applicable category headers}'."""
    for r_idx, row in enumerate(data_rows, start=1):
        item = str(row[0]).replace("\n", " ").strip() if row[0] else ""
        if not item:
            continue
        applies_to = []
        for c in range(1, min(len(row), len(headers))):
            if row[c] == "✓":
                applies_to.append(headers[c])
        if not applies_to:
            continue  # row has no checkmarks → not meaningful, skip
        text = f"{item}: {', '.join(applies_to)}"
        yield {
            "chunk_id": f"{pdf_name}_p{page_idx}_t{t_idx}_r{r_idx}",
            "source_file": pdf_filename,
            "page": page_idx,
            "chunk_type": "table_row",
            "parameter": item,
            "text": text,
            "raw_cells": [str(c) if c is not None else "" for c in row],
            "headers": headers,
        }


def emit_paramvalue_chunks(data_rows, pdf_name, pdf_filename, page_idx, t_idx):
    """For Parameter|Guideline style tables: pipe-joined cells."""
    for r_idx, row in enumerate(data_rows, start=1):
        cells = [
            str(c).replace("\n", " ").strip()
            for c in row
            if c not in (None, "")
        ]
        if not cells:
            continue
        text = " | ".join(cells)
        yield {
            "chunk_id": f"{pdf_name}_p{page_idx}_t{t_idx}_r{r_idx}",
            "source_file": pdf_filename,
            "page": page_idx,
            "chunk_type": "table_row",
            "parameter": str(row[0]) if row[0] else None,
            "text": text,
            "raw_cells": [str(c) if c is not None else "" for c in row],
        }


# ---------- Prose helpers (unchanged from v2) ----------

def collect_table_words(tables):
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


# ---------- Main per-PDF processing ----------

def process_pdf(pdf_path):
    pdf_name = pdf_path.stem
    pdf_filename = pdf_path.name
    # Headers persist across tables within a document so multi-page tables
    # whose header only appears on page 0 still get meaningful chunks later.
    carried_headers = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for t_idx, raw_table in enumerate(tables):
                cleaned = drop_empty_columns(raw_table)
                if not cleaned:
                    continue

                header_rows, data_rows = split_header_data(cleaned)
                n_cols = len(cleaned[0])

                # Update carried headers if this table has its own
                if header_rows:
                    carried_headers = reconstruct_headers(header_rows, n_cols)

                headers = carried_headers or [f"col_{i}" for i in range(n_cols)]

                forward_fill_col0(data_rows)
                # Drop rows that are entirely empty after ffill
                data_rows = [
                    r for r in data_rows if any(v not in (None, "") for v in r)
                ]
                if not data_rows:
                    continue

                if is_checkbox_table(data_rows):
                    yield from emit_checkbox_chunks(
                        data_rows, headers, pdf_name, pdf_filename, page_idx, t_idx
                    )
                else:
                    yield from emit_paramvalue_chunks(
                        data_rows, pdf_name, pdf_filename, page_idx, t_idx
                    )

            # Prose chunks: extract page text and filter against table content
            table_words = collect_table_words(tables)
            text = page.extract_text() or ""
            for p_idx, paragraph in enumerate(split_prose(text)):
                if len(paragraph) < MIN_PROSE_CHARS:
                    continue
                if is_table_residue(paragraph, table_words):
                    continue
                yield {
                    "chunk_id": f"{pdf_name}_p{page_idx}_para{p_idx}",
                    "source_file": pdf_filename,
                    "page": page_idx,
                    "chunk_type": "prose",
                    "text": paragraph,
                }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {RAW_DIR}")

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

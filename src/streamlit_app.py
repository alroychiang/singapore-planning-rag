"""
Streamlit UI for the Singapore Planning RAG system.

Displays:
- Question input
- Generated answer (grounded in retrieved chunks, with [N] citations)
- Retrieved chunks: source, page, similarity score, chunk_type, full text

Uses:
- query() from query.py for retrieval
- build_context / client / MODEL / PROMPT_TEMPLATE from generate.py for
  augmentation and generation

Run:
  streamlit run src/streamlit_app.py

Requires:
- Chroma index built (run src/index.py first)
- GEMINI_API_KEY in .env at repo root
"""

import json
from pathlib import Path

import streamlit as st

from generate import query
from generate import build_context, run_llm, MODEL, PROMPT_TEMPLATE, BACKEND

QUERIES_PATH = Path("data/eval/queries.jsonl")


def load_example_queries():
    """Read tried-and-tested example queries from the eval set."""
    queries = []
    if QUERIES_PATH.exists():
        with open(QUERIES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
    return queries

# ---------- Page setup ----------

st.set_page_config(
    page_title="Singapore Planning RAG",
    page_icon="🏙️",
    layout="wide",
)

# ---------- Info popover (fixed bottom-left) ----------

st.markdown(
    """
    <style>
    /* Trim the large default empty space at the top of the page */
    .block-container {
        padding-top: 2rem;
    }
    /* Pin the bottom bar (holding both popovers) to the bottom-left */
    .st-key-bottom_bar {
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        width: auto;
        z-index: 1000;
    }
    /* Lay the two popover buttons out in a row */
    .st-key-bottom_bar [data-testid="stVerticalBlock"] {
        flex-direction: row;
        gap: 0.5rem;
        width: auto;
    }
    .st-key-bottom_bar div[data-testid="stPopover"] button {
        width: auto;
        min-width: 0;
        padding: 0.25rem 0.75rem;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="bottom_bar"):
    with st.popover("info", help="System info & known limitations"):
        st.markdown("**System**")
        st.markdown(
            f"""
            - **Embedding model**: `all-MiniLM-L6-v2`
            - **LLM backend**: `{BACKEND}`
            - **LLM model**: `{MODEL}`
            - **Vector DB**: Chroma (local, persistent)
            - **Corpus**: 20 URA planning PDFs, ~770 chunks
            """
        )
        st.divider()
        st.markdown("**Known limitations**")
        st.markdown(
            """
            Surfaced by the retrieval eval (`reports/eval_summary.md`):

            - **Prose chunking too coarse.** Master Plan definitions live in
              large paragraph chunks; single-definition queries can miss.
            - **Extraction bug in one GFA table.** Header reconstruction
              failed for the Bay Windows row, producing a chunk that reads
              "Bay Windows: col_3".
            - **Domain terminology.** General-purpose embeddings weakly
              connect "residential" ↔ "HDB estates" / "landed housing".

            Fixes in the roadmap: bbox-based prose extraction, hybrid
            keyword+vector retrieval, upgrade to a stronger embedding model.
            """
        )

    with st.popover("queries", help="Tried-and-tested example queries"):
        st.markdown("**Example queries**")
        st.caption("Copy any of these to try the system out.")
        for q in load_example_queries():
            st.markdown(f"- {q['query']}")

# ---------- Main UI (two columns: query left, references right) ----------

left, spacer, right = st.columns([1, 0.1, 1])

with left:
    st.subheader("Singapore Planning RAG", anchor=False)
    st.caption(
        "Ask questions about URA planning guidelines from official URA published documents"
    )

    question = st.text_input(
        "Question",
        placeholder="e.g. What is the road buffer for an expressway?",
        label_visibility="collapsed",
    )

    k = st.slider(
        "No. of references",
        min_value=3,
        max_value=15,
        value=5,
    )

with right:
    st.subheader("Retrieved references", anchor=False)
    st.caption(
        "Each reference shows its source document, page, and similarity score"
    )

# ---------- Retrieve + generate on submit ----------

if question:
    with st.spinner("Retrieving relevant chunks..."):
        results = query(question, k=k)

    with st.spinner(f"Generating answer with {MODEL}..."):
        context = build_context(results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = run_llm(prompt)

    # ----- Answer (left column) -----
    with left:
        st.subheader("Answer", anchor=False)
        with st.container(border=True):
            st.markdown(answer)

    # ----- Retrieved chunks (right column) -----
    with right:
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (id_, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            similarity = 1 - dist
            header = (
                f"[{i}]  {meta['source_file']} · page {meta['page']} · "
                f"similarity {similarity:.3f}"
            )
            with st.expander(header):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Chunk ID**: `{id_}`")
                col2.markdown(f"**Type**: {meta['chunk_type']}")
                if meta.get("parameter"):
                    st.markdown(f"**Parameter**: {meta['parameter']}")
                st.markdown("**Text**:")
                st.code(doc, language="text")

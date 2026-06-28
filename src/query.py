"""
Query the Chroma collection built by index.py.

Embeds a question using the SAME sentence-transformer model used at index
time, then retrieves the top-K most similar chunks via cosine similarity.

No LLM generation yet — this script only does retrieval. It exists to
validate that the right chunks come back for representative questions
BEFORE we wire up generation in the next step.

Test queries cover three retrieval scenarios:
- prose chunks (Master Plan written statement)
- table_row chunks with parameter forward-fill (Summary handbooks)
- table_row chunks with checkbox-style cells (GFA Summary)
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "ura_planning"
MODEL_NAME = "all-MiniLM-L6-v2"  # MUST match the model used in embed.py


# Load resources once at module level so repeated queries are fast
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(MODEL_NAME)


def query(question: str, k: int = 5):
    """Embed a question and return the top-K nearest chunks from Chroma."""
    q_vec = model.encode(question)
    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
    )
    return results


def pretty_print(question: str, results: dict):
    """Flatten Chroma's nested result structure and print readably."""
    print(f"\n{'=' * 70}")
    print(f"Q: {question}")
    print(f"{'=' * 70}")

    # Chroma returns lists-of-lists because it can handle batched queries.
    # We only sent one query, so everything lives in index [0].
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (id_, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        # Convert cosine distance (0 = identical) to similarity (1 = identical)
        similarity = 1 - dist
        text_preview = doc.replace("\n", " ")[:180]
        print(
            f"\n[{rank}] similarity={similarity:.3f}  "
            f"({meta['source_file']}, p{meta['page']}, {meta['chunk_type']})"
        )
        print(f"    {text_preview}")


def main():
    test_questions = [
        "What is the maximum plot ratio for residential developments?",
        "Is a void deck included as GFA?",
        "What is the road buffer for an expressway?",
    ]

    for q in test_questions:
        results = query(q, k=5)
        pretty_print(q, results)


if __name__ == "__main__":
    main()

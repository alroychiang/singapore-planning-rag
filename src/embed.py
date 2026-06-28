"""
Embed chunks using sentence-transformers.

Reads data/processed/chunks.jsonl and writes data/processed/embeddings.npy
(numpy array of shape [N, 384]). Row order matches the chunks.jsonl line
order, so chunks[i] corresponds to embeddings[i].

Model: all-MiniLM-L6-v2
- 384-dimensional embeddings
- ~80MB download on first run, cached afterwards
- Good general-purpose semantic similarity, runs fine on CPU
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    # Load chunks
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")

    # Load model (first run downloads weights from HuggingFace)
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    # for sanity check
    print(f"Embedding dimension: {dim}")

    # Embed all chunk texts in one batched call
    # gets the text field only. (data values)
    texts = [c["text"] for c in chunks]
    print("Embedding...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

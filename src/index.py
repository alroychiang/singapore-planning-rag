"""
Build a Chroma vector index from chunks + embeddings.

Reads:
- data/processed/chunks.jsonl
- data/processed/embeddings.npy

Writes:
- data/processed/chroma/  (persistent Chroma database)

Each chunk becomes a Chroma record with:
- id:        chunk_id
- embedding: 384-dim vector from embeddings.npy
- document:  the chunk's text field (what gets returned at query time)
- metadata:  source_file, page, chunk_type, parameter (filterable at query)

The collection is deleted and rebuilt on each run — keeps the script
idempotent and avoids stale-state bugs while iterating.
"""

import json
from pathlib import Path

import chromadb
import numpy as np

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
# client
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "ura_planning"


def main():
    # Load chunks
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # Load embeddings and verify alignment
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(chunks) == len(embeddings), (
        f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
    )
    print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

    # Persistent Chroma client — stores to disk so the index survives restarts
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate the collection from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    # "cosine" matches the normalized vectors from sentence-transformers
    # collection == connection to vector database. many collections 
    # for different collections in one Client
    collection = client.create_collection(
        name=COLLECTION_NAME,
        # l2 euclidean, or ip inner product, or cosine (default for sentence transformers)
        metadata={"hnsw:space": "cosine"},
    )

    # Build the payload. Chroma metadata values must be str/int/float/bool,
    # so we coerce None -> "" for the parameter field.
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_file": c["source_file"],
            "page": c["page"],
            "chunk_type": c["chunk_type"],
            "parameter": c.get("parameter") or "",
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Added {collection.count()} chunks to collection '{COLLECTION_NAME}'")
    print(f"Persisted to {CHROMA_DIR}")

    # Peek at one record so you can see what landed
    peek = collection.peek(limit=1)
    print("\nFirst record in collection:")
    print(f"  id:       {peek['ids'][0]}")
    print(f"  document: {peek['documents'][0][:100]}...")
    print(f"  metadata: {peek['metadatas'][0]}")


if __name__ == "__main__":
    main()

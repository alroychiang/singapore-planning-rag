"""
RAG generation: retrieve chunks, augment prompt, generate grounded answer.

Pipeline per question:
1. Reuse query() from query.py to get top-K chunks from Chroma
2. Build an augmented prompt: instructions + numbered context + question
3. Send to Gemini, return the answer

The LLM is instructed to:
- Answer ONLY from provided context (no training-data hallucinations)
- Cite sources by their [N] number in the context block
- Say "I cannot find this in the provided documents" if the answer isn't there

This last rule is the most important guardrail. It mirrors the "N/A instead
of hallucinate" pattern from the Apple extraction pipeline.

Requires: GOOGLE_API_KEY in environment (or in a .env file at repo root)

takes query output (chunks) & inserts into a structured prompt before feeding Gemini API to produce a more human readable response
"""

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads .env if present

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit(
        "GOOGLE_API_KEY not set. Add it to .env or export it in your shell."
    )

# Gemini
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

# Retrieval
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "ura_planning"
MODEL_NAME = "all-MiniLM-L6-v2"

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(COLLECTION_NAME)
embed_model = SentenceTransformer(MODEL_NAME)


PROMPT_TEMPLATE = """You are an assistant that answers questions about Singapore Urban Redevelopment Authority (URA) planning guidelines.

Answer the question using ONLY the information in the context below. If the answer cannot be determined from the context, respond exactly: "I cannot find this information in the provided documents."

When citing information, refer to sources by their bracketed number, like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""

def query(question: str, k: int = 15):
    """Embed a question and return the top-K nearest chunks from Chroma."""
    q_vec = embed_model.encode(question)
    return collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
    )

# source of data, ground truth only for the LLM to look through. (creating database for LLM)
# LLM doesnt look through corpus manually
def build_context(results: dict) -> str:
    """Format Chroma results into a numbered context block for the LLM."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    lines = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = f"{meta['source_file']}, p{meta['page']}, {meta['chunk_type']}"
        lines.append(f"[{i}] ({source})\n{doc}")
    return "\n\n".join(lines)


def generate(question: str, k: int = 15) -> dict:
    """Retrieve k chunks, ask Gemini to answer using them. Returns answer + sources."""
    # uses function from query.py. Custom questions found in this .py file
    results = query(question, k=k)
    context = build_context(results)
    # Finalize a prompt with Database Ground truth, user question and guardrails for LLM
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # gemini 2.5 flash API client library
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    distances = results["distances"][0]
    return {
        "question": question,
        "answer": response.text,
        "sources": [
            {
                "n": i + 1,
                "source_file": meta["source_file"],
                "page": meta["page"],
                "similarity": round(1 - dist, 3),
                "text": doc[:200],
            }
            for i, (doc, meta, dist) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], distances)
            )
        ],
    }


def pretty_print(result: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"Q: {result['question']}")
    print(f"{'=' * 72}")
    print(f"\n{result['answer']}\n")
    print("Sources used:")
    for src in result["sources"]:
        print(
            f"  [{src['n']}] {src['source_file']}, p{src['page']}  similarity={src['similarity']:.3f}"
            f"\n      {src['text'].replace(chr(10), ' ')[:160]}..."
        )


def main():
    # test_questions = [
    #     "What is the road buffer for an expressway?",
    #     "Is a void deck included as GFA?",
    #     "What is the maximum plot ratio for residential developments?",
    #     "What is the minimum unit size for B1 industrial developments?",
    # ]

    test_questions = [
        "What is the maximum plot ratio for residential developments?",
    ]

    for q in test_questions:
        result = generate(q, k=15)
        pretty_print(result)


if __name__ == "__main__":
    main()

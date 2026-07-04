"""
RAG generation: retrieve chunks, augment prompt, generate grounded answer.

Pipeline per question:
1. Reuse query() from query.py to get top-K chunks from Chroma
2. Build an augmented prompt: instructions + numbered context + question
3. Send to LLM (Ollama by default, Gemini via LLM_BACKEND=gemini), return the answer

The LLM is instructed to:
- Answer ONLY from provided context (no training-data hallucinations)
- Cite sources by their [N] number in the context block
- Say "I cannot find this in the provided documents" if the answer isn't there

Env vars:
- LLM_BACKEND:    "ollama" (default) or "gemini"
- OLLAMA_MODEL:   defaults to "qwen3:4b"
- OLLAMA_HOST:    defaults to "http://localhost:11434"
- GOOGLE_API_KEY: required if LLM_BACKEND=gemini
- GEMINI_MODEL:   defaults to "gemini-2.5-flash-lite"
"""

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads .env if present

BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# Retrieval
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "ura_planning"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(COLLECTION_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


PROMPT_TEMPLATE = """You are an assistant that answers questions about Singapore Urban Redevelopment Authority (URA) planning guidelines.

Answer the question using ONLY the information in the context below. If the answer cannot be determined from the context, respond exactly: "I cannot find this information in the provided documents."

When citing information, refer to sources by their bracketed number, like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""


# ---------- Backend dispatch ----------

if BACKEND == "gemini":
    from google import genai

    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        sys.exit("GOOGLE_API_KEY not set. Add it to .env or unset LLM_BACKEND.")

    MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    _client = genai.Client(api_key=API_KEY)

    def run_llm(prompt: str) -> str:
        response = _client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        return response.text

elif BACKEND == "ollama":
    import ollama

    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    # defaults as safety net
    MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    _client = ollama.Client(host=OLLAMA_HOST)

    def run_llm(prompt: str) -> str:
        response = _client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

else:
    sys.exit(f"Unknown LLM_BACKEND: '{BACKEND}'. Use 'ollama' or 'gemini'.")


# ---------- Retrieval ----------

def query(question: str, k: int = 15):
    """Embed a question and return the top-K nearest chunks from Chroma."""
    q_vec = embed_model.encode(question)
    return collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
    )


def build_context(results: dict) -> str:
    """Format Chroma results into a numbered context block for the LLM."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    lines = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = f"{meta['source_file']}, p{meta['page']}, {meta['chunk_type']}"
        lines.append(f"[{i}] ({source})\n{doc}")
    return "\n\n".join(lines)


# ---------- Public API ----------

def generate(question: str, k: int = 15) -> dict:
    """Retrieve k chunks, ask LLM to answer using them. Returns answer + sources."""
    results = query(question, k=k)
    context = build_context(results)
    print(context)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = run_llm(prompt)

    distances = results["distances"][0]
    return {
        "question": question,
        "answer": answer,
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
    print(f"Backend: {BACKEND}  |  Model: {MODEL}\n")
    test_questions = [
        "What is the road buffer for an expressway?",
        "Who is the current CEO of URA?",
        "What is the plot ratio for my plot of land at 123 Sengkang Drive?",
    ]

    for q in test_questions:
        result = generate(q, k=15)
        pretty_print(result)


if __name__ == "__main__":
    main()
A RAG system that answers questions pertaining Singapore URA planning documents

![alt text](image.png)

Run demo locally:
install docker application @ docker.com
pull image "docker pull alroychiang/singapore-planning-rag"
run image "docker run -p 8501:8501 --env-file .env alroychiang/singapore-planning-rag"
open browser and paste link "http://localhost:8501"

URA RAG system allows users to get answers based on official documents and sources without worrying for model hallucination. Answers are grounded to a fixed private database with citation/provenance where users can verify references accurately. Knowledge base updatable in the future without needing to retrain any models or expensive fine-tuning required to keep answers current.

architecture diagram (ASCII): URA PDF documents in data -> chunking (extract.py) -> vectorize with sentence transformer (embed.py) -> index to vectorDB (chroma) -> retrieve chunk into LLM (qwen or gemini) for structured answer

INDEX PIPELINE (one-time build):

```
┌─────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐
│  URA PDFs   │──▶│ extract.py │──▶│  embed.py  │──▶│ index.py │──▶│  Chroma DB │
│ (data/raw)  │   │            │   │            │   │          │   │(persistent)│
└─────────────┘   └────────────┘   └────────────┘   └──────────┘   └────────────┘
```

QUERY PIPELINE (per user question):

```
  User query
      │
      ▼
┌────────────┐        ┌────────────┐        ┌────────────────────┐
│  query.py  │───────▶│  Chroma DB │───────▶│    generate.py     │
│            │        │            │        │                    │
└────────────┘        └────────────┘        │  Ollama (qwen3:4b) │
                                            │        or          │
                                            │    Gemini API      │
                                            └─────────┬──────────┘
                                                      │
                                                      ▼
                                          Grounded answer + citations
```

Design decisions
Decided Chroma vectorDB because lightweight, easy to run locally, free (Pinecone is enterprise level, Weaviate supports hybrid search (lexical + semantic))

Decided sentence transformer model: all-MiniLM-L6-v2, small, light weight, reproducible in another machine for small projects

ollama: small scale, ease of set up, self hosted model for working on sensitive government documents. Consider vLLM should scale rise.

Manually created a custom extract/chunking script instead of using LlamaIndex nor Langchain's chunking tool as they chunk by character counts (500) and are less capable of handling table data. Although they do have advance table chunking services, they are paid at scale or beyond the free tier

for queries that aren't found within the corpus, LLM responds with "I cannot find this information in the provided documents." to prevent hallucination from training data.

Eval.py evaluates RAG pipeline with 15 questions targeted at possible pipeline weaknesses. This file may be re-run again should corpus change, chunking changed, vectorization model change, vectorDB change or augmentation LLM change. Requires a Golden Dataset with handwritten queries and handpicked ideal answer chunks "queries.jsonl" where 'k' is the number of results you choose to retrieve from the total no. of results your vectorDB returns. 'Relevant chunks' are chunks found within your golden dataset.

Three metrics were used
    - Precision@k: (no. of relevant chunks in results) / k
    - Recall@k: (no. of relevant chunks in results) / (total relevant chunks in golden set)
    - Mean Reciprocal Rank: 1/ (index position of first relevant result found) 

EVALUATION:
Precision@k measures the amount of noise retrieved, "why does the pipeline pick up irrelevant chunks". Precision@5 average was 0.25, meaning only 25% of our retrieved chunks are relevant to the posed question. Our golden data only has 1-2 ideal chunks. Thus the higher 'k' you set (asking the model to retrieve 'k' number of 'correct' chunks), the lower your Precision@k's percentage will seem.

Recall@k measures the coverage of our retrieval, "are we able to pick up all relevant chunks". Recall@5 average was 0.806, which means 80% of all our golden dataset's chunks were retrieved by the model to answer queries.

MRR measures the average position the first relevant result shows up in your retrieval list. 
MRR average was 0.609, meaning the first relevant chunk appears on average at rank ~1.6 (1/0.609 ≈ 1.64)

Guardrails and fact reliability to corpus:
"What is the maximum plot ratio for HDB residential estates?" 
answer: After reviewing the provided context, the maximum plot ratio for HDB residential estates is specified in document [1] as "up to 1.6" for "Within HDB estates and in areas with GPR more than 1.4". However, the context does not provide a single, unconditional maximum plot ratio for all HDB residential estates without specific conditions (e.g., "areas with GPR more than 1.4").
The context shows that different documents ([1], [2], [3]) indicate varying maximum plot ratios (1.6, 1.4, 1.4) for HDB estates under specific conditions. Since the question asks for the maximum plot ratio for HDB residential estates in general (without conditions), and the context does not explicitly state a universal maximum value applicable to all HDB estates, the answer cannot be definitively determined from the provided information.
Therefore, the response is:
I cannot find this information in the provided documents.
analysis: states seemingly close results but is sure that isnt the answer. cite references and gives guardrail response instead.

Limitations
analysis: for questions that are hard to find in corpus, qwen takes longer to answer Runaway rambling (not refusal nor real answer)
Possible solution: caps the number of output words model is allowed to produce to shorten runaway rambling response == less time it takes, or reduce 'k' chunks retrieved, or tighten augmentation prompt

"What is the definition of plot ratio in the Master Plan?" fails to retrieve an answer
answer: "I cannot find this information in the provided documents."
MP25WrittenStatement_p5_para0 in corpus states "1.1.13 “plot ratio” means the ratio between the floor area of the building and site area." which is the right answer.
analysis: The chunk consisting this answer was embedded into a vector with seven other different definitions, causing the vector to be identified as a "generic definition" block, which was not picked up by ChromaDB at all when cosine similarity matching.
sol: chunk smaller, possibly by numbered sections instead of /n. Amend extraction.py

"Are bay windows counted toward GFA?"
answer: "I cannot find this information in the provided documents.", Bay Windows: col_3, Summary_GFA_p0_t3_r4
analysis: col oriented table columns cause chunking to miss headers, CCI questions are row wise

Attribution
The corpus consists of publicly available planning documents from the Singapore Urban Redevelopment Authority (URA), including the Master Plan Written Statement, Development Control Handbooks, and Urban Design Guidelines.
These documents are © Singapore Government and remain the property of URA. They are downloaded on demand via download.sh from URA's public website and are not redistributed as part of this repository. Users must fetch them directly from source. This project is a technical demonstration and is not affiliated with or endorsed by URA.
For official planning guidance, refer to www.ura.gov.sg.
A RAG system that answers questions pertaining Singapore URA planning documents

![alt text](image.png)

Run demo locally:
install docker application @ docker.com
pull image "docker pull <your-dockerhub-username>/singapore-planning-rag"
run image "docker run -p 8501:8501 --env-file .env <your-dockerhub-username>/singapore-planning-rag"
open browser and paste link "http://localhost:8501"

URA RAG system allows users to get answers based on official documents and sources without worrying for model hallucination. Answers are grounded to a fixed private database with citation/provenance where users can verify references accurately. Knowledge base updatable in the future without needing to retrain any models or expensive fine turning required to keep answers current.

architecture digram (ASCII please help): URA PDF documents in data -> chunking (extract_new.py) -> vectorize with sentence transformer (embed.py) -> index to vectorDB (chroma) -> retrieve chunk into LLM (qwen or gemini) for structured answer

Design decisions
Decided Chroma vectorDB because lightweight, easy to run locally, free (Pinecone is enterprise level, Weaviate is lexical + semantic searches)

Decided sentence transformer model: all-MiniLM-L6-v2, small, light weight, reproducible in another machine for small projects

ollama: small scale, ease of set up, self hosted model for working on sensitive government documents. Consider vLLM should scale rise.

Manually created a custom extract/chunking script instead of using LlamaIndex nor Langchain's chunking tool as they chunk by character counts (500) and are less capable of handling table data. Although they do have advance table chunking services, they are paid at scale or beyond the free tier

for queries that aren't found within the corpus, LLM responds with "I cannot find this information in the provided documents." to curb hallucination to respond with its own tranining data.

Eval.py evaluates RAG pipeline with 15 questions targetted at possible pipeline weaknesses. This file may be re-runned again should corpus change, chunking changed, vectorization model change, vectorDB change or augmentation LLM change. Requires a Golden Dataset with handwritten queries and handpicked ideal answer chunks "queries.jsonl" where 'k' is the number of results you choose to retrieve from the total no. of results your vectorDB returns. 'Relevant chunks' are chunks found within your golden dataset.

Three metrics were used
    - Precision@k: (no. of relevant chunks in results) / k
    - Recall@k: (no. of relevant chunks in results) / (total relevant chunks in golden set)
    - Mean Reciprocal Rank: 1/ (index position of first relevant result found) 

Precision@k measures the amount of noise retrieved, "why does the pipeline pick up irrelevant chunks"
Recall@k measures the coverage of our retrieval, "are we able to pick up all relevant chunks". Note: Recall@k is only meaningful when k is reasonably close in scale to the number of relevant chunks you're trying to catch. 
MRR measures the average position the first relevant result shows up in your retrieval list.

Evalute my program: results and findings and improvements.... TODO
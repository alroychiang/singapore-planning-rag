# Dockerfile — Singapore Planning RAG
#
# Ships the Streamlit UI, the RAG pipeline code, and the pre-built Chroma index.
# Uses the Gemini LLM backend by default (no Ollama service needed in the container).
#
# Build:
#   docker build -t singapore-planning-rag .
#
# Run:
#   docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key_here singapore-planning-rag
#
# Or with an .env file:
#   docker run -p 8501:8501 --env-file .env singapore-planning-rag
#
# Then open http://localhost:8501

# a mini operating system w/ python installed
# FROM sets base image of the container, docker things build on top of this
FROM python:3.11-slim

# sets working directory inside container
WORKDIR /app

# Install Python dependencies first (cache-friendly: only re-runs on requirements change)
# installs sentence transformer library (no weights installed yet)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model into the image.
# Adds ~90MB but eliminates first-request latency.
# this code triggers the actual model weights download from Hugging face
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code and the pre-built retrieval index
COPY src/ ./src/
COPY data/processed/chroma/ ./data/processed/chroma/

# Default configuration: Gemini backend (Ollama isn't in the container)
ENV LLM_BACKEND=gemini
ENV GEMINI_MODEL=gemini-2.5-flash-lite

# container's port. to be connected to mac's port on docker run time
# just a note saying this container uses port 8501 internally
EXPOSE 8501

# --server.address=0.0.0.0 allows container to have cnnections from outside itself other than processes running within
# container.
# "--server.headless=true" prevents the PC from opening the application using a browser window within the container
# user has to manually go local host URL
# (required so the host machine can reach the container)
CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

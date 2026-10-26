# Hugging Face Spaces (Docker SDK) / any Docker host. Listens on port 7860.
FROM python:3.11-slim

# faiss needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only torch first (the default wheel is ~2 GB with CUDA we do not need)
COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

# Bake both models into the image so the first user does not wait for a download
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app/ui.py", "--server.port=7860", "--server.address=0.0.0.0"]

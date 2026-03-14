# Kani TTS / GM Voice Studio API
FROM python:3.12-slim

WORKDIR /app

# Build/runtime packages for audio and PDF features used by the refactor app.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install deps in two steps to avoid pip resolution-too-deep.
COPY requirements-core.txt requirements-server.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-server.txt && \
    pip install --no-cache-dir -r requirements-rag.txt

COPY . .

# routes_legacy imports fitz at module load, so keep PyMuPDF explicit in the image.
RUN pip install --no-cache-dir "pymupdf>=1.24.0"

EXPOSE 7862
ENV PORT=7862
CMD ["python", "server.py"]

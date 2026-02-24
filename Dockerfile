# Kani TTS / GM Voice Studio API
FROM python:3.12-slim

WORKDIR /app

# Install deps in two steps to avoid pip resolution-too-deep
COPY requirements-core.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-server.txt

COPY . .
EXPOSE 7862
ENV PORT=7862
CMD ["python", "server.py"]

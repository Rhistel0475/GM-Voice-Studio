"""
Co-DM document ingestion CLI.
Loads PDFs or plain text files, chunks them, embeds with OpenAI, and upserts to Pinecone.

Usage:
    python ingest.py --file path/to/rulebook.pdf --doc_type rulebook
    python ingest.py --dir path/to/docs/ --doc_type session
    python ingest.py --file notes.txt --doc_type faction

doc_type choices: rulebook | faction | bestiary | session
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DOC_TYPES = ("rulebook", "faction", "bestiary", "session")


def _load_documents(path: Path, doc_type: str) -> list:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        logging.warning("Skipping unsupported file type: %s", path.name)
        return []

    docs = loader.load()
    # Attach doc_type and normalise source to filename only
    for doc in docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["source"] = path.name
    return docs


def ingest(paths: list[Path], doc_type: str) -> int:
    """Ingest a list of files. Returns total chunks upserted."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    # Load env / config
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set. Add it to your .env file.")
    if not PINECONE_API_KEY:
        sys.exit("PINECONE_API_KEY is not set. Add it to your .env file.")

    # Load and chunk documents
    all_docs = []
    for p in paths:
        docs = _load_documents(p, doc_type)
        logging.info("Loaded %d pages from %s", len(docs), p.name)
        all_docs.extend(docs)

    if not all_docs:
        logging.warning("No documents loaded — nothing to ingest.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    logging.info("Split into %d chunks", len(chunks))

    # Embed and upsert
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    logging.info("Upserted %d chunks to Pinecone index '%s'", len(chunks), PINECONE_INDEX_NAME)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Co-DM RAG knowledge base.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="PATH", help="Single PDF or text file to ingest")
    group.add_argument("--dir", metavar="DIR", help="Directory — ingest all .pdf and .txt files inside")
    parser.add_argument(
        "--doc_type",
        choices=DOC_TYPES,
        default="rulebook",
        help="Document category tag attached to every chunk (default: rulebook)",
    )
    args = parser.parse_args()

    if args.file:
        paths = [Path(args.file)]
    else:
        d = Path(args.dir)
        if not d.is_dir():
            sys.exit(f"Not a directory: {d}")
        paths = [p for p in d.iterdir() if p.suffix.lower() in (".pdf", ".txt", ".md")]
        if not paths:
            sys.exit(f"No PDF or text files found in {d}")

    for p in paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    total = ingest(paths, args.doc_type)
    print(f"Done. {total} chunks ingested as doc_type='{args.doc_type}'.")


if __name__ == "__main__":
    main()

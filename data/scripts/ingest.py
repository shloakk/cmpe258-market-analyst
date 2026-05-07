"""
Ingestion script: loads the JSON corpus, embeds each document using
OpenAI text-embedding-3-small, and saves a FAISS index to data/index/.

Usage:
    python data/scripts/ingest.py

Environment variables required:
    OPENAI_API_KEY
"""

import json
import os
import pathlib

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

CORPUS_PATH = pathlib.Path(__file__).parent.parent / "corpus" / "sample_docs.json"
INDEX_DIR = pathlib.Path(__file__).parent.parent / "index"


def load_corpus(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def doc_to_langchain(record: dict) -> Document:
    """Convert a normalized corpus record to a LangChain Document."""
    content = f"{record['title']}\n\n{record['description']}"
    metadata = {
        "title": record["title"],
        "source_url": record["source_url"],
        "tags": record.get("tags", []),
        "publish_date": record.get("publish_date", ""),
    }
    return Document(page_content=content, metadata=metadata)


def build_index(records: list[dict]) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [doc_to_langchain(r) for r in records]
    print(f"Embedding {len(docs)} documents...")
    index = FAISS.from_documents(docs, embeddings)
    return index


def save_index(index: FAISS, directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index.save_local(str(directory))
    print(f"FAISS index saved to {directory}/")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    records = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(records)} documents from corpus.")

    index = build_index(records)
    save_index(index, INDEX_DIR)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()

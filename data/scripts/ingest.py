"""
Ingestion script: loads the JSON corpus, embeds each document using
sentence-transformers/all-MiniLM-L6-v2 (local, no API key required),
and saves a FAISS index to data/index/.

Usage:
    python data/scripts/ingest.py
"""

import json
import pathlib

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"
INDEX_DIR = pathlib.Path(__file__).parent.parent / "index"


def load_corpus(corpus_dir: pathlib.Path) -> list[dict]:
    """
    Load and merge all *.json files in corpus_dir into a single flat list.

    Each file must contain a JSON array of record dicts with keys:
    title, description, source_url, tags, publish_date.

    Deduplicates by source_url so records present in multiple shards
    (e.g. sample_docs.json and startups.json) are not embedded twice.
    The same URL scraped by two different source functions would otherwise
    produce duplicate embeddings that skew retrieval scores.

    Args:
        corpus_dir: Directory containing one or more corpus shard JSON files.

    Returns:
        Deduplicated list of record dicts, sorted by shard filename for
        deterministic ordering across runs.

    Raises:
        FileNotFoundError: If corpus_dir contains no *.json files.
    """
    shard_files = sorted(corpus_dir.glob("*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No *.json files found in {corpus_dir}")

    seen: set[str] = set()
    records: list[dict] = []
    for path in shard_files:
        with open(path) as f:
            shard = json.load(f)
        for record in shard:
            url = record.get("source_url", "")
            if url not in seen:
                seen.add(url)
                records.append(record)
    return records


def doc_to_langchain(record: dict) -> Document:
    """
    Convert a normalized corpus record to a LangChain Document.

    Args:
        record: Dict with keys title, description, source_url, tags, publish_date.

    Returns:
        LangChain Document with title+description as page_content and
        remaining fields as metadata.
    """
    content = f"{record['title']}\n\n{record['description']}"
    metadata = {
        "title": record["title"],
        "source_url": record["source_url"],
        "tags": record.get("tags", []),
        "publish_date": record.get("publish_date", ""),
    }
    return Document(page_content=content, metadata=metadata)


def build_index(records: list[dict]) -> FAISS:
    """
    Embed all corpus records and construct a FAISS vector index.

    Args:
        records: List of normalized corpus record dicts.

    Returns:
        FAISS index loaded with all document embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [doc_to_langchain(r) for r in records]
    print(f"Embedding {len(docs)} documents...")
    index = FAISS.from_documents(docs, embeddings)
    return index


def save_index(index: FAISS, directory: pathlib.Path) -> None:
    """
    Persist the FAISS index to disk.

    Args:
        index:     Populated FAISS index.
        directory: Target directory; created if absent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    index.save_local(str(directory))
    print(f"FAISS index saved to {directory}/")


def main() -> None:
    """
    Entry point: loads all corpus shards, builds and saves index.

    Raises:
        FileNotFoundError: If no corpus JSON files are found.
    """
    records = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(records)} documents from corpus.")

    index = build_index(records)
    save_index(index, INDEX_DIR)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()

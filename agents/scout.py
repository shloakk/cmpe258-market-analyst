"""
Scout Agent: embeds the user query and retrieves the top-k most relevant
documents from the FAISS vector index built by data/scripts/ingest.py.
"""

import pathlib
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = pathlib.Path(__file__).parent.parent / "data" / "index"
DEFAULT_TOP_K = 10  # 10 is appropriate for a 200+ doc corpus; was 5 for the 10-doc seed


class ScoutAgent:
    """Retrieves documents relevant to a market research query."""

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        self.top_k = top_k
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def run(self, query: str) -> tuple[list[dict], dict]:
        """
        Args:
            query: Natural language market research query.

        Returns:
            Tuple of (docs, stats) where docs is a list of document dicts with
            keys: title, description, source_url, tags, publish_date, snippet,
            score; and stats contains retrieval latency.
        """
        t0 = time.perf_counter()
        results = self.index.similarity_search_with_score(query, k=self.top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        docs = []
        for doc, score in results:
            docs.append(
                {
                    "title": doc.metadata.get("title", ""),
                    "source_url": doc.metadata.get("source_url", ""),
                    "tags": doc.metadata.get("tags", []),
                    "publish_date": doc.metadata.get("publish_date", ""),
                    "snippet": doc.page_content,
                    "score": float(score),
                }
            )
        stats = {"latency_ms": latency_ms, "docs_retrieved": len(docs)}
        return docs, stats

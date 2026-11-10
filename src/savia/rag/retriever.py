"""Similarity search with a relevance threshold."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from savia.rag.models import Chunk, RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Retrieval parameters."""

    k: int = 8
    max_distance: float = 0.4
    include_archived: bool = False


class PolicyRetriever:
    """Retrieve corpus chunks relevant to a customer question."""

    def __init__(self, store: VectorStore, config: RetrieverConfig) -> None:
        """Store dependencies. No I/O in the constructor."""
        self._store = store
        self._config = config

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return up to k chunks with score <= max_distance, best first.

        Returns an empty list (never raises) when nothing is relevant enough:
        the caller turns an empty result into an "I don't know" answer.
        """
        chunks: list[tuple[Document, float]] = self._store.similarity_search_with_score(
            query=query,
            k=self._config.k,
        )

        return [
            RetrievedChunk(
                chunk=Chunk(**document.metadata, text=document.page_content), score=score
            )
            for document, score in chunks
            if score <= self._config.max_distance
        ]

"""Vector index construction and loading."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.errors import NotFoundError
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from savia.rag.chunking import ChunkingConfig
from savia.rag.models import Chunk

logger = logging.getLogger(__name__)

MAX_LENGTH_COLLECTION_NAME = 62
_ID_LOOKUP_BATCH_SIZE = 500


def _to_documents(chunks: list[Chunk]) -> list[Document]:
    return [
        Document(
            id=chunk.chunk_id, page_content=chunk.text, metadata=chunk.model_dump(exclude={"text"})
        )
        for chunk in chunks
    ]


def _existing_ids(store: Chroma, ids: list[str]) -> set[str]:
    """Return the subset of `ids` already present in the collection (batched lookup)."""
    found: set[str] = set()
    for start in range(0, len(ids), _ID_LOOKUP_BATCH_SIZE):
        batch = ids[start : start + _ID_LOOKUP_BATCH_SIZE]
        found.update(store.get(ids=batch, include=[])["ids"])
    return found


def collection_name(embedding_model: str, chunking: ChunkingConfig) -> str:
    """Return a collection name that encodes everything that affects vectors.

    Example: "supportops__text-embedding-3-small__chunk-v1".
    """
    name = f"savia__{embedding_model}__chunk-{chunking.version}"
    if len(name) > MAX_LENGTH_COLLECTION_NAME:
        over = len(name) - MAX_LENGTH_COLLECTION_NAME
        name = f"savia__{embedding_model[0:-over]}__chunk-{chunking.version}"
    return name.replace(".", "-").replace("/", "-")


def build_index(
    chunks: list[Chunk],
    embeddings: Embeddings,
    persist_dir: Path,
    name: str,
) -> tuple[VectorStore, list[Chunk]]:
    """Create or update the index. Idempotent: re-running ingests nothing new.

    Uses chunk_id as the vector id so that re-ingestion upserts instead of
    duplicating. Uses cosine distance explicitly.

    Precondition: `chunk_id` must be derived from the chunk content (e.g. include a
    content hash). Chunks whose id already exists in the collection are skipped
    and not re-embedded, so an id that stays the same while the text changes would
    leave stale content in the index.

    Args:
        chunks: Chunks to index.
        embeddings: Embedding model used to vectorize the chunks.
        persist_dir: Directory where Chroma persists the collection.
        name: Collection name (see `collection_name`).

    Returns:
        The vector store backed by the collection.

    Raises:
        ValueError: If two chunks share the same `chunk_id`.
    """
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk_id found in the chunks to index")

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_configuration=CreateCollectionConfiguration({"hnsw": {"space": "cosine"}}),
    )

    already_indexed = _existing_ids(store, ids)
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in already_indexed]
    if new_chunks:
        store.add_documents(
            _to_documents(new_chunks),
            ids=[chunk.chunk_id for chunk in new_chunks],
        )
    return store, new_chunks


class IndexNotFoundError(RuntimeError):
    """Raised when the vector index is missing or empty."""


def load_index(embeddings: Embeddings, persist_dir: Path, name: str) -> VectorStore:
    """Open an existing index, failing loudly if the collection is missing or empty."""
    if not persist_dir.is_dir():
        raise IndexNotFoundError(f"Directory not found: {persist_dir.name}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=name)
    except NotFoundError as exc:
        raise IndexNotFoundError(
            f"Collection '{name}' not found in {persist_dir}. Run the ingestion first."
        ) from exc

    if collection.count() == 0:
        raise IndexNotFoundError(f"Collection '{name}' is empty in {persist_dir}.")

    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=name,
        create_collection_if_not_exists=False,
    )

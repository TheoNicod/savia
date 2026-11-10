"""Round-trip tests for the Chroma index (build -> persist -> reload).

Deterministic fake embeddings are used on purpose: these tests check the
*plumbing* (counts, ids, idempotence, persistence, error paths), not retrieval
relevance. Relevance is evaluated later with a real embedding model.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from savia.rag.chunking import ChunkingConfig
from savia.rag.index import IndexNotFoundError, build_index, collection_name, load_index
from savia.rag.models import Chunk

pytestmark = pytest.mark.integration

_EMBEDDING_SIZE = 32


class SpyEmbeddings(Embeddings):
    """Delegate to another embedding model and record every text sent to it.

    This lets a test prove that nothing was (re-)embedded, which a simple
    vector count cannot show: embedding calls cost money with a real provider.
    """

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def chunks() -> list[Chunk]:
    """Hand-built chunks, so these tests do not depend on `chunk_document`."""
    return [
        Chunk(
            chunk_id=f"policy-refund@2026-01#{i:03d}/v1",
            doc_id="policy-refund",
            section=f"Politique > Section {i}",
            text=f"Texte fictif du chunk numéro {i}.",
            position=i,
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def embeddings() -> SpyEmbeddings:
    return SpyEmbeddings(DeterministicFakeEmbedding(size=_EMBEDDING_SIZE))


@pytest.fixture
def persist_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def index_name() -> str:
    return collection_name("fake-embedding", ChunkingConfig())


def _open_collection(persist_dir: Path, name: str) -> chromadb.Collection:
    """Open the collection with a fresh native client (independent of LangChain).

    Going through a separate client also proves the data was really persisted.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=name)


# ---------------------------------------------------------------- the 3 core checks
def test_build_index_stores_one_vector_per_chunk(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)

    assert _open_collection(persist_dir, index_name).count() == len(chunks)


def test_stored_ids_are_the_chunk_ids(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)

    stored_ids = set(_open_collection(persist_dir, index_name).get(include=[])["ids"])

    # Sets, not lists: Chroma does not guarantee any ordering.
    assert stored_ids == {chunk.chunk_id for chunk in chunks}


def test_build_index_is_idempotent(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)
    ids_after_first_run = set(_open_collection(persist_dir, index_name).get(include=[])["ids"])
    embeddings.embedded_texts.clear()

    build_index(chunks, embeddings, persist_dir, index_name)
    collection = _open_collection(persist_dir, index_name)

    assert collection.count() == len(chunks)
    assert set(collection.get(include=[])["ids"]) == ids_after_first_run
    assert embeddings.embedded_texts == [], "re-running must not re-embed anything"


# ---------------------------------------------------------------- build_index: more behaviours
def test_build_index_only_embeds_new_chunks(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks[:3], embeddings, persist_dir, index_name)
    embeddings.embedded_texts.clear()

    build_index(chunks, embeddings, persist_dir, index_name)

    assert embeddings.embedded_texts == [chunk.text for chunk in chunks[3:]]
    assert _open_collection(persist_dir, index_name).count() == len(chunks)


def test_build_index_rejects_duplicate_chunk_ids(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    duplicated = [*chunks, chunks[0]]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        build_index(duplicated, embeddings, persist_dir, index_name)

    # Fail fast: validation happens before any side effect on disk.
    assert not persist_dir.exists()
    assert embeddings.embedded_texts == []


def test_build_index_uses_cosine_distance(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)

    configuration = _open_collection(persist_dir, index_name).configuration

    assert configuration["hnsw"]["space"] == "cosine"


def test_same_id_with_changed_text_is_not_reindexed(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    """Characterization test: documents a known limitation of the current design.

    `chunk_id` does not include a content hash, so a chunk whose text changed
    but whose id did not is skipped and the OLD text stays in the index.
    If ids become content-derived, rewrite this test to expect the new text.
    """
    build_index(chunks, embeddings, persist_dir, index_name)
    edited = chunks[0].model_copy(update={"text": "Texte modifié, même identifiant."})

    build_index([edited, *chunks[1:]], embeddings, persist_dir, index_name)

    stored = _open_collection(persist_dir, index_name).get(
        ids=[edited.chunk_id], include=["documents"]
    )
    assert stored["documents"] == [chunks[0].text]


# ---------------------------------------------------------------- load_index
def test_load_index_roundtrip_returns_stored_chunk_with_metadata(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)

    store = load_index(embeddings, persist_dir, index_name)
    # Same text => same fake vector => the exact chunk is the nearest neighbour.
    # This checks the round trip, NOT relevance.
    results = store.similarity_search(chunks[2].text, k=1)

    assert len(results) == 1
    found = results[0]
    assert found.page_content == chunks[2].text
    assert found.metadata["chunk_id"] == chunks[2].chunk_id
    assert found.metadata["doc_id"] == chunks[2].doc_id
    assert found.metadata["section"] == chunks[2].section
    assert found.metadata["position"] == chunks[2].position
    assert "text" not in found.metadata, "the text belongs in page_content, not in metadata"


def test_load_index_fails_when_directory_is_missing(
    embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    with pytest.raises(IndexNotFoundError):
        load_index(embeddings, persist_dir, index_name)


def test_load_index_fails_when_collection_is_missing(
    chunks: list[Chunk], embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    build_index(chunks, embeddings, persist_dir, index_name)

    with pytest.raises(IndexNotFoundError, match="not found"):
        load_index(embeddings, persist_dir, "another-collection")


def test_load_index_fails_when_collection_is_empty(
    embeddings: SpyEmbeddings, persist_dir: Path, index_name: str
) -> None:
    persist_dir.mkdir(parents=True)
    chromadb.PersistentClient(path=str(persist_dir)).create_collection(name=index_name)

    with pytest.raises(IndexNotFoundError, match="empty"):
        load_index(embeddings, persist_dir, index_name)

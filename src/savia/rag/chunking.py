"""Deterministic chunking of corpus documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from savia.rag.models import Chunk, SourceDocument


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters. Changing them requires a full re-index."""

    chunk_size: int = 800  # characters
    chunk_overlap: int = 120
    version: str = "v1"  # bump when any parameter changes


def make_chunk_id(doc_id: str, doc_version: str, position: int, config: ChunkingConfig) -> str:
    """Return a stable, human-readable chunk id.

    Example: "policy-refund@2026-01#003/v1".
    Same inputs must ALWAYS return the same id: it is the key for upserts
    and for citations stored in eval datasets.
    """
    return f"{doc_id.strip()}@{doc_version.strip()}#{position:03d}/{config.version}"


def _build_section(
    metadata: Mapping[str, str],
    keys: tuple[str, ...] = ("h1", "h2", "h3", "h4"),
    separator: str = ">",
) -> str:
    """Concatène les titres présents dans les métadonnées, dans l'ordre des niveaux.

    Les niveaux absents, vides ou non textuels sont ignorés.
    """
    parts = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return separator.join(parts)


def chunk_document(doc: SourceDocument, config: ChunkingConfig) -> list[Chunk]:
    """Split a document by markdown headings, then by size with overlap.

    Each chunk carries the heading path of its section so that a citation
    can say *where* in the document the information comes from.
    """
    header_docs = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
        return_each_line=False,
    ).split_text(doc.body)

    final_docs = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    ).split_documents(header_docs)

    chunks: list[Chunk] = []

    for i, final_doc in enumerate(final_docs, start=1):
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, doc.version, i, config),
                doc_id=doc.doc_id,
                section=_build_section(final_doc.metadata),
                text=final_doc.page_content,
                position=i,
            )
        )
    return chunks

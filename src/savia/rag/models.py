"""Typed contracts for the RAG pipeline."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocCategory = Literal["faq", "cgv", "policy", "subscription", "legal"]
DocStatus = Literal["active", "archived"]


class SourceDocument(BaseModel):
    """A validated corpus document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str
    category: DocCategory
    version: str
    effective_date: date
    status: DocStatus = "active"
    body: str = Field(min_length=1)


class Chunk(BaseModel):
    """A slice of a document, ready to embed."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # deterministic, see chunking.py
    doc_id: str
    section: str  # nearest markdown heading path, e.g. "Politique > Plafond"
    text: str
    position: int  # order within the document


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever with its relevance score in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)

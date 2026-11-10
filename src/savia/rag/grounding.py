"""Grounded answer generation and citation validation."""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from savia.rag.models import RetrievedChunk
from savia.rag.prompts import RAG_PROMPT, RAG_PROMPT_VERSION, format_context

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """A pointer from a claim to the chunk that supports it."""

    chunk_id: str
    quote: str = Field(max_length=300, description="Short supporting excerpt from the chunk.")


class GroundedAnswer(BaseModel):
    """Structured output requested from the LLM."""

    is_answerable: bool = Field(
        description="False if the provided context does not contain the answer."
    )
    answer: str
    citations: list[Citation]


class GroundingError(Exception):
    """Raised when an answer cites something that was not retrieved."""


NO_ANSWER_MESSAGE = (
    "Je n'ai pas trouvé cette information dans nos documents. "
    "Je transmets votre question à un conseiller."
)


def validate_grounding(answer: GroundedAnswer, retrieved: list[RetrievedChunk]) -> GroundedAnswer:
    """Check that every citation points to a retrieved chunk and quotes it faithfully.

    Rules:
        - is_answerable=True requires at least one citation.
        - Every citation.chunk_id must belong to `retrieved`.
        - Every citation.quote must appear in the cited chunk text
          (after Markdown stripping, typographic and whitespace normalization;
          punctuation at both ends of the quote is ignored).

        - is_answerable=False requires no citation (a refusal cites nothing).
        - A quote must not be empty (an empty string "appears" in every chunk).

    Raises:
        GroundingError: Listing every violation, so the trace is actionable.
    """
    texts = {item.chunk.chunk_id: _normalize(item.chunk.text) for item in retrieved}
    violations: list[str] = []

    if answer.is_answerable and not answer.citations:
        violations.append("is_answerable=True but no citation")
    if not answer.is_answerable and answer.citations:
        violations.append("is_answerable=False but citations were given")

    for index, citation in enumerate(answer.citations):
        where = f"citation[{index}] ({citation.chunk_id!r})"
        quote = _normalize_quote(citation.quote)
        if citation.chunk_id not in texts:
            violations.append(f"{where}: chunk_id was not retrieved")
        elif not quote:
            violations.append(f"{where}: empty quote")
        elif quote not in texts[citation.chunk_id]:
            violations.append(f"{where}: quote not found in the cited chunk")

    if violations:
        raise GroundingError("; ".join(violations))
    return answer


# Markdown that the LLM drops when quoting: inline emphasis/code markers
# (**bold**, *italic*, __bold__, _italic_, ~~strike~~, `code`) and heading markers.
# `_` is only stripped at a word boundary, so snake_case identifiers survive.
_MARKDOWN_INLINE = re.compile(r"\*+|`+|~~|(?<!\w)_+|_+(?!\w)")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)


# Typographic variants the LLM may swap freely: unify them on both sides.
_TYPOGRAPHY = str.maketrans(
    {
        "’": "'",  # ’
        "‘": "'",  # ‘
        "ʼ": "'",  # ʼ
        "«": '"',  # «
        "»": '"',  # »
        "“": '"',  # “
        "”": '"',  # ”
        "–": "-",  # –
        "—": "-",  # —
        "‑": "-",  # non-breaking hyphen
        " ": " ",  # non-breaking space
        " ": " ",  # narrow non-breaking space (French « ; : ! ? »)
        "…": "...",  # …
    }
)

# Punctuation and quote marks tolerated at the edges of a quote only (sentence cut
# short with an added period, quote wrapped in quotation marks...). Inner punctuation
# is still compared, so a distorted figure or word is still caught.
_QUOTE_EDGES = " .,;:!?\"'()-"


def _normalize(text: str) -> str:
    """Make chunk text and quotes comparable.

    Strips Markdown formatting, unifies typographic variants (apostrophes,
    quotation marks, dashes, non-breaking spaces), then collapses every run
    of whitespace into one space. Applied to both sides, so the model may quote
    with or without the formatting (`**14 jours**` and `14 jours` both match).
    """
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_INLINE.sub("", text)
    text = text.translate(_TYPOGRAPHY)
    return " ".join(text.split())


def _normalize_quote(quote: str) -> str:
    """`_normalize`, plus trimming of the punctuation at both ends of the quote."""
    return _normalize(quote).strip(_QUOTE_EDGES)


def _refusal() -> GroundedAnswer:
    return GroundedAnswer(is_answerable=False, answer=NO_ANSWER_MESSAGE, citations=[])


def answer_from_corpus(
    question: str,
    retrieved: list[RetrievedChunk],
    model: BaseChatModel,
) -> GroundedAnswer:
    """Generate a grounded answer, or a refusal when context is missing or invalid.

    - Empty `retrieved` → refusal WITHOUT calling the LLM (cheaper, safer).
    - Grounding failure → refusal + the error is logged, never shown to the customer.
    """
    if not retrieved:
        return _refusal()

    chain = RAG_PROMPT | model.with_structured_output(GroundedAnswer)
    answer = chain.invoke({"context": format_context(retrieved), "question": question})
    if not isinstance(answer, GroundedAnswer):
        logger.warning("Unexpected structured output type: %s", type(answer).__name__)
        return _refusal()
    if not answer.is_answerable:
        return _refusal()

    try:
        return validate_grounding(answer, retrieved)
    except GroundingError as exc:
        # Never log the question: it may contain customer PII.
        logger.warning(
            "Grounding failed (prompt %s), answer replaced by a refusal: %s",
            RAG_PROMPT_VERSION,
            exc,
        )
        return _refusal()

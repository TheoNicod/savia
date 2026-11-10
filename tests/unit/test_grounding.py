"""Tests for citation validation and grounded answer generation."""

from __future__ import annotations

import logging

import pytest
from langchain.messages import AIMessage

from savia.rag.grounding import (
    NO_ANSWER_MESSAGE,
    Citation,
    GroundedAnswer,
    GroundingError,
    answer_from_corpus,
    validate_grounding,
)
from savia.rag.models import Chunk, RetrievedChunk

################################
# Helpers
################################

_REFUND_ID = "policy-refund@2026-01#001/v1"
_SUB_ID = "subscription@2026-01#002/v1"
_INVENTED_ID = "policy-refund@2026-01#099/v1"

_REFUND_TEXT = (
    "Le client dispose d'un délai de 30 jours après la livraison\npour demander un remboursement."
)
_SUB_TEXT = "La résiliation est possible avec un préavis de 10 jours avant le prélèvement."


def _retrieved(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=chunk_id.split("@", 1)[0],
            section="Politique > Remboursement",
            text=text,
            position=1,
        ),
        score=score,
    )


@pytest.fixture
def retrieved() -> list[RetrievedChunk]:
    return [_retrieved(_REFUND_ID, _REFUND_TEXT), _retrieved(_SUB_ID, _SUB_TEXT)]


def _answer(
    *citations: tuple[str, str], answerable: bool = True, text: str = "Réponse."
) -> GroundedAnswer:
    return GroundedAnswer(
        is_answerable=answerable,
        answer=text,
        citations=[Citation(chunk_id=cid, quote=quote) for cid, quote in citations],
    )


def _structured_reply(answer: GroundedAnswer) -> AIMessage:
    """Scripted reply as `with_structured_output(GroundedAnswer)` expects it (tool call)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "GroundedAnswer", "args": answer.model_dump(), "id": "call_1"}],
    )


################################
# validate_grounding
################################


def test_valid_citation_is_accepted(retrieved):
    answer = _answer((_REFUND_ID, "30 jours après la livraison"))

    assert validate_grounding(answer, retrieved) == answer


def test_several_valid_citations_across_chunks_are_accepted(retrieved):
    answer = _answer(
        (_REFUND_ID, "30 jours après la livraison"),
        (_SUB_ID, "préavis de 10 jours"),
    )

    assert validate_grounding(answer, retrieved) == answer


def test_invented_chunk_id_is_rejected(retrieved):
    answer = _answer((_INVENTED_ID, "30 jours après la livraison"))

    with pytest.raises(GroundingError, match="#099"):
        validate_grounding(answer, retrieved)


def test_distorted_quote_is_rejected(retrieved):
    """A single changed figure ("15" instead of "30") must be caught."""
    answer = _answer((_REFUND_ID, "15 jours après la livraison"))

    with pytest.raises(GroundingError):
        validate_grounding(answer, retrieved)


def test_quote_attributed_to_the_wrong_chunk_is_rejected(retrieved):
    """The quote exists in the context, but not in the chunk it claims to come from."""
    answer = _answer((_REFUND_ID, "préavis de 10 jours"))

    with pytest.raises(GroundingError):
        validate_grounding(answer, retrieved)


def test_answerable_without_citation_is_rejected(retrieved):
    answer = _answer(answerable=True)

    with pytest.raises(GroundingError):
        validate_grounding(answer, retrieved)


def test_well_formed_refusal_is_accepted(retrieved):
    refusal = _answer(answerable=False, text="")

    assert validate_grounding(refusal, retrieved) == refusal


def test_quote_matches_after_whitespace_normalization(retrieved):
    """The chunk has a line break, the quote has spaces: same text, must pass."""
    answer = _answer((_REFUND_ID, "après   la livraison pour demander\tun remboursement"))

    assert validate_grounding(answer, retrieved) == answer


_MD_ID = "cgv-v3@2026-01#006/v1"
_MD_TEXT = (
    "## Article 5 — Droit de rétractation légal\n"
    "Le consommateur dispose d'un délai légal de **14 jours** à compter de la "
    "réception pour se rétracter, *sans justification*, via le `formulaire`."
)


@pytest.mark.parametrize(
    "quote",
    [
        "délai légal de 14 jours à compter de la réception",  # formatting dropped
        "délai légal de **14 jours** à compter",  # formatting kept
        "se rétracter, sans justification, via le formulaire",
        "Article 5 — Droit de rétractation légal",  # heading without "##"
    ],
)
def test_quote_matches_regardless_of_markdown_formatting(quote):
    answer = _answer((_MD_ID, quote))

    assert validate_grounding(answer, [_retrieved(_MD_ID, _MD_TEXT)]) == answer


_CGV_TEXT = (
    "## Article 5 — Droit de rétractation légal\n"
    "Le consommateur dispose d'un délai légal de **14 jours** à compter de la réception "
    "pour se rétracter, sans justification, conformément au Code de la consommation ; "
    "il renvoie le produit au plus tard **14 jours** après cette information."
)


@pytest.mark.parametrize(
    "quote",
    [
        # Real model output: sentence cut short, closing period added.
        "Le consommateur dispose d'un délai légal de 14 jours à compter de la réception "
        "pour se rétracter, sans justification.",
        "« pour se rétracter, sans justification »",  # wrapped in quotation marks
        "Le consommateur dispose d’un délai légal",  # curly apostrophe
        "Article 5 - Droit de rétractation",  # plain dash instead of em dash
        "Code de la consommation ; il renvoie",  # regular space before ";"
        "...pour se rétracter…",  # ellipsis on both ends
    ],
)
def test_quote_matches_despite_edge_punctuation_and_typography(quote):
    answer = _answer((_MD_ID, quote))

    assert validate_grounding(answer, [_retrieved(_MD_ID, _CGV_TEXT)]) == answer


@pytest.mark.parametrize(
    "quote",
    [
        "délai légal de 14 jours. à compter",  # punctuation added INSIDE the quote
        "délai légal de 15 jours à compter de la réception.",  # distorted figure
        ".",  # only punctuation: empty once trimmed
    ],
)
def test_inner_changes_are_still_rejected(quote):
    answer = _answer((_MD_ID, quote))

    with pytest.raises(GroundingError):
        validate_grounding(answer, [_retrieved(_MD_ID, _CGV_TEXT)])


def test_distorted_quote_is_still_rejected_in_markdown_chunk():
    answer = _answer((_MD_ID, "délai légal de 30 jours à compter de la réception"))

    with pytest.raises(GroundingError):
        validate_grounding(answer, [_retrieved(_MD_ID, _MD_TEXT)])


def test_any_citation_is_rejected_when_nothing_was_retrieved():
    answer = _answer((_REFUND_ID, "30 jours après la livraison"))

    with pytest.raises(GroundingError):
        validate_grounding(answer, [])


def test_error_lists_every_violation(retrieved):
    """One invented id + one distorted quote: both must appear in the error."""
    answer = _answer(
        (_INVENTED_ID, "30 jours après la livraison"),
        (_SUB_ID, "préavis de 5 jours"),
    )

    with pytest.raises(GroundingError) as exc_info:
        validate_grounding(answer, retrieved)

    message = str(exc_info.value)
    assert _INVENTED_ID in message
    assert _SUB_ID in message


################################
# answer_from_corpus
################################


def _assert_is_refusal(answer: GroundedAnswer) -> None:
    assert answer.is_answerable is False
    assert answer.answer == NO_ANSWER_MESSAGE
    assert answer.citations == []


def test_empty_retrieval_refuses_without_calling_the_model(fake_model_factory):
    """An empty script proves it: any model call would raise."""
    model = fake_model_factory(messages=iter([]))

    answer = answer_from_corpus("Quel est le délai de remboursement ?", [], model)

    _assert_is_refusal(answer)


def test_grounded_model_answer_is_returned(fake_model_factory, retrieved):
    expected = _answer(
        (_REFUND_ID, "30 jours après la livraison"),
        text="Vous disposez de 30 jours après la livraison.",
    )
    model = fake_model_factory(messages=iter([_structured_reply(expected)]))

    answer = answer_from_corpus("Quel est le délai de remboursement ?", retrieved, model)

    assert answer == expected


def test_ungrounded_model_answer_becomes_a_logged_refusal(fake_model_factory, retrieved, caplog):
    hallucinated = _answer(
        (_INVENTED_ID, "15 jours après la livraison"),
        text="Vous disposez de 15 jours.",
    )
    model = fake_model_factory(messages=iter([_structured_reply(hallucinated)]))

    with caplog.at_level(logging.WARNING):
        answer = answer_from_corpus("Quel est le délai de remboursement ?", retrieved, model)

    _assert_is_refusal(answer)
    assert _INVENTED_ID not in answer.answer  # the error never reaches the customer
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_model_refusal_is_returned_as_refusal(fake_model_factory, retrieved):
    model = fake_model_factory(
        messages=iter([_structured_reply(_answer(answerable=False, text=""))])
    )

    answer = answer_from_corpus("Quelle est la couleur du logo ?", retrieved, model)

    _assert_is_refusal(answer)

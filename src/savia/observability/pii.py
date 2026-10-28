"""PII redaction utilities. Pure functions, no I/O, no external dependency."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Final

logger = logging.getLogger(__name__)

MAX_DEPTH = 50
REDACTION_FAILED = "[REDACTION_FAILED]"
MAX_DEPTH_MARKER = "[MAX_DEPTH_MARKER]"
CYCLE_MARKER = "[REDACTED_CYCLE]"

EMAIL_RE: Final = re.compile(
    r"""
    (?<![A-Za-z0-9._%+-])
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*
    \.[A-Za-z]{2,}
    """,
    re.VERBOSE,
)

# Pays (2 lettres) + clé (2 chiffres) + 11 à 30 caractères, espaces facultatifs.
# Majuscules uniquement (format normalisé) pour limiter les faux positifs.
IBAN_RE: Final = re.compile(
    r"""
    (?<![A-Za-z0-9])
    [A-Z]{2}\d{2}
    (?:[ \u00A0]?[A-Z0-9]){11,30}
    (?![A-Za-z0-9])
    """,
    re.VERBOSE,
)

# Visa / Mastercard (16 chiffres) et American Express (15 chiffres),
# séparateur espace ou tiret facultatif avant chaque chiffre.
CARD_RE: Final = re.compile(
    r"""
    (?<!\d)
    (?:
        (?:
            4\d{3}                                          # Visa
          | 5[1-5]\d{2}                                     # Mastercard 51-55
          | 2(?:22[1-9]|2[3-9]\d|[3-6]\d{2}|7[01]\d|720)    # Mastercard 2221-2720
        )
        (?:[ -]?\d){12}
      | 3[47]\d{2}(?:[ -]?\d){11}                           # American Express
    )
    (?!\d)
    """,
    re.VERBOSE,
)

_OCTET: Final = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
IPV4_RE: Final = re.compile(rf"(?<!\d)(?<!\d\.){_OCTET}(?:\.{_OCTET}){{3}}(?!\d)(?!\.\d)")

# Commence par '+' (international) ou '0' (national / "00" international),
# puis 8 à 15 chiffres. Séparateurs tolérés : espace, point, tiret ;
# parenthèses tolérées autour d'un groupe, ex. "+1 (415) 555-2671", "+33 (0)6 ...".
# Chaque « unité » se termine par exactement un chiffre : pas de backtracking
# exponentiel, et une parenthèse fermante finale n'est jamais absorbée.
PHONE_RE: Final = re.compile(
    r"""
    (?<![\w+])
    [+0]
    (?:\)?[\s.\-]?\(?\d){8,15}
    """,
    re.VERBOSE,
)


def redact_text(text: str) -> str:
    """Replace every PII occurrence with a typed placeholder, e.g. `<EMAIL>`, `<PHONE>`.

    Typed placeholders (not just `***`) keep traces debuggable: you still
    see that the user *gave* an email, without seeing which one.
    """
    if not isinstance(text, str):
        raise TypeError(f"redact_text attend une str, reçu : {type(text).__name__}")

    try:
        res = text
        for pattern, tag in (
            (IBAN_RE, "<IBAN>"),  # IBAN avant carte/téléphone
            (CARD_RE, "<CARD>"),
            (EMAIL_RE, "<EMAIL>"),
            (PHONE_RE, "<PHONE>"),
            (IPV4_RE, "<IP>"),
        ):
            res = pattern.sub(tag, res)
        return res
    except Exception as exc:
        logger.warning("redact_text failed: %s", type(exc).__name__)
        return REDACTION_FAILED


def redact_value(value: object) -> object:
    """Recursively redact strings inside dicts, lists and tuples.

    Returns a new structure of the same shape; the input is never mutated.
    Dict keys are kept as-is, only values are redacted. Non-string leaves
    (int, None, custom objects...) are returned unchanged.

    Must never raise: a crash in the redaction layer must not crash the agent.
    Failures are fail-closed, nothing unredacted leaks:

    - a string whose redaction fails is replaced by `REDACTION_FAILED`,
      the rest of the structure is preserved;
    - any other error (e.g. a tuple subclass that cannot be rebuilt)
      replaces the whole value by `REDACTION_FAILED`;
    - nesting beyond `MAX_DEPTH` is replaced by `MAX_DEPTH_MARKER`;
    - a container referencing one of its ancestors is replaced by
      `CYCLE_MARKER`.
    """
    try:
        return _r_redact(value, depth=0, path=set())
    except Exception as exc:
        logger.warning("_r_redact failed: %s", type(exc).__name__)
        return REDACTION_FAILED


def _r_redact(value: object, depth: int, path: set[int]) -> object:
    """Recursive worker behind `redact_value`.

    Args:
        value: current node to redact.
        depth: nesting level of `value`, 0 at the root.
        path: `id()` of the containers on the current branch, from the root
            to `value`. Used to detect cycles; ids are removed on the way
            back up, so a container shared by two sibling branches is
            redacted twice rather than flagged as a cycle.

    Leaf redaction errors are caught here (fail-closed per string). Errors
    while rebuilding a container propagate to `redact_value`.
    """
    if isinstance(value, str):
        try:
            return redact_text(value)
        except Exception as exc:
            logger.warning("redact_text failed: %s", type(exc).__name__)
            return REDACTION_FAILED

    if not isinstance(value, (list, tuple, dict)):
        return value

    # Garde-fous
    if depth >= MAX_DEPTH:
        return MAX_DEPTH_MARKER
    obj_id = id(value)
    if obj_id in path:
        return CYCLE_MARKER

    path.add(obj_id)
    try:
        if isinstance(value, dict):
            return {k: _r_redact(v, depth + 1, path) for k, v in value.items()}

        items = [_r_redact(v, depth + 1, path) for v in value]
        if isinstance(value, list):
            return items
        if hasattr(value, "_fields"):  # namedtuple
            # mypy voit `type[tuple]` (un seul itérable) ; un namedtuple
            # prend ses champs en arguments positionnels.
            ctor: Callable[..., object] = type(value)
            return ctor(*items)
        return tuple(items)
    finally:
        path.discard(obj_id)

"""Load and validate the markdown corpus (front-matter + body)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from savia.rag.models import SourceDocument

logger = logging.getLogger(__name__)

_DELIMITER = "---"


class CorpusValidationError(Exception):
    """Un document du corpus (ou le dossier lui-même) est invalide.

    Le message nomme le fichier et le champ fautif, jamais le contenu du document.
    """

    def __init__(self, path: Path, field: str | None, reason: str) -> None:
        """Construit l'erreur.

        Args:
            path: Fichier (ou dossier) en cause.
            field: Champ fautif, ou None si l'erreur ne concerne pas un champ précis.
            reason: Explication courte, sans valeur de donnée.
        """
        self.path = path
        self.field = field
        self.reason = reason
        where = f"champ '{field}'" if field else "document"
        super().__init__(f"{path.name} - {where} : {reason}")


def _split_frontmatter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    """Sépare l'en-tête YAML du corps Markdown.

    Returns:
        Le dictionnaire de métadonnées et le corps du document.

    Raises:
        CorpusValidationError: En-tête absent, non fermé, YAML invalide ou pas un mapping.
    """
    lines = raw.removeprefix("\ufeff").splitlines()  # tolère un BOM éventuel
    if not lines or lines[0].strip() != _DELIMITER:
        raise CorpusValidationError(path, None, "front-matter absent (premier '---' manquant)")

    closing = next(
        (i for i, line in enumerate(lines[1:], start=1) if line.strip() == _DELIMITER),
        None,
    )
    if closing is None:
        raise CorpusValidationError(path, None, "front-matter non fermé (second '---' manquant)")

    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise CorpusValidationError(path, None, "YAML invalide dans le front-matter") from exc

    if not isinstance(metadata, dict):
        raise CorpusValidationError(path, None, "le front-matter doit être de la forme clé: valeur")

    body = "\n".join(lines[closing + 1 :]).strip()
    return metadata, body


def _load_document(path: Path) -> SourceDocument:
    """Read one Markdown file and validate it as a `SourceDocument`.

    Args:
        path: Markdown file with a YAML front-matter header.

    Returns:
        The validated document, with the Markdown body in its `body` field.

    Raises:
        CorpusValidationError: If the file is not UTF-8, the front-matter is
            malformed, or the metadata fails model validation (the error
            reports the first offending field).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusValidationError(path, None, "Fichier non encodé en UTF-8") from exc

    metadata, body = _split_frontmatter(raw, path)

    try:
        return SourceDocument.model_validate({**metadata, "body": body})
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or None
        raise CorpusValidationError(path, field, first["msg"]) from exc


def load_corpus(path: Path) -> list[SourceDocument]:
    """Load every Markdown document of a corpus directory.

    Files are read in alphabetical order (non-recursive `*.md`). Documents whose
    status is "archived" are skipped.

    Args:
        path: Directory containing the corpus `.md` files.

    Returns:
        The validated, non-archived documents.

    Raises:
        CorpusValidationError: If `path` is not a directory, contains no `.md`
            file, has an invalid document, or two documents share a `doc_id`.
    """
    if not path.is_dir():
        raise CorpusValidationError(
            path, field=None, reason="Le chemin d'accès ne mène pas à un dossier"
        )

    corpus_files = sorted(file for file in path.glob("*.md") if file.is_file())

    if not corpus_files:
        raise CorpusValidationError(path=path, field=None, reason="Aucun fichier .md trouvé")

    document_list: list[SourceDocument] = []
    seen: dict[str, Path] = {}

    for file in corpus_files:
        document = _load_document(file)
        if hasattr(document, "status") and document.status == "archived":
            logger.info("%s est une archive. Skip.", document.doc_id)
            continue
        if document.doc_id in seen:
            raise CorpusValidationError(
                file, "doc_id", f"doublon avec {seen[document.doc_id].name}"
            )
        seen[document.doc_id] = file
        document_list.append(document)
    return document_list

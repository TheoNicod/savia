"""Ingestion script: load the corpus, split it into chunks and index it in Chroma.

Usage:
    python scripts/ingest.py [--corpus DIR] [--index DIR]
"""

import argparse
import logging
from pathlib import Path

from savia.config import get_settings
from savia.llm.factory import build_embeddings
from savia.rag.chunking import ChunkingConfig, chunk_document
from savia.rag.corpus import CorpusValidationError, load_corpus
from savia.rag.index import build_index, collection_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus"
DEFAULT_INDEX = ROOT / "data" / "index"

CHUNKING = ChunkingConfig()

logger = logging.getLogger("savia.ingest")


def run_ingest(corpus_dir: Path, index_dir: Path) -> None:
    """Index the corpus: load, chunk, embed, then write to Chroma.

    The collection name depends on the embeddings model and the chunking
    configuration, so incompatible indexes are never mixed.

    Args:
        corpus_dir: Directory containing the corpus documents to index.
        index_dir: Directory where the Chroma index is persisted.

    Raises:
        CorpusValidationError: If the corpus is invalid.
    """
    docs = [doc for doc in load_corpus(corpus_dir)]
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, CHUNKING)]
    logger.info("Chunking: %d documents, %d chunks", len(docs), len(chunks))

    settings = get_settings()
    name = collection_name(settings.embeddings_model, CHUNKING)
    _, new_chunks = build_index(chunks, build_embeddings(settings), index_dir, name)
    logger.info("Indexing: %d chunks -> %s", len(new_chunks), name)


def main() -> int:
    """Command-line entry point.

    Returns:
        0 on success, 1 if the corpus is invalid.
    """
    parser = argparse.ArgumentParser(description="Indexe le corpus dans Chroma.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_ingest(args.corpus, args.index)
    except CorpusValidationError as exc:
        logger.error("Corpus invalide : %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

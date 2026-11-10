"""Offline evaluation and tuning of the policy retriever.

Loads the persisted vector index and a retrieval dataset (``evals/datasets``),
then computes hit rate, MRR and false-positive rate, or grid-searches the
retriever parameters (``k`` and ``max_distance``).
"""

import json
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from savia.config import Settings, get_settings
from savia.llm.factory import build_embeddings
from savia.rag.chunking import ChunkingConfig
from savia.rag.index import collection_name, load_index
from savia.rag.models import RetrievedChunk
from savia.rag.retriever import PolicyRetriever, RetrieverConfig

ROOT = Path(__file__).resolve().parents[1]
DATASETS_PATH = ROOT / "evals" / "datasets"
INDEX_PATH = ROOT / "data" / "index"

CHUNKING_CONFIG = ChunkingConfig()


def _load_dataset(filename: str) -> list[dict]:
    """Load an evaluation dataset from the datasets directory.

    Args:
        filename: Name of the dataset file, relative to ``evals/datasets``.

    Returns:
        The parsed JSON content of the file.
    """
    with open(DATASETS_PATH / filename, encoding="utf-8") as file:
        return json.load(file)


def _get_retriever(settings: Settings, retriever_config: RetrieverConfig) -> PolicyRetriever:
    """Build a ``PolicyRetriever`` on top of the persisted index.

    The collection name is derived from the embeddings model and the chunking
    configuration, so it matches the index built with the same parameters.

    Args:
        settings: Application settings (used to build the embeddings).
        retriever_config: Retrieval parameters (e.g. ``k``, ``max_distance``).

    Returns:
        A retriever bound to the loaded vector store.
    """
    vs = load_index(
        embeddings=build_embeddings(settings),
        persist_dir=INDEX_PATH,
        name=collection_name(settings.embeddings_model, CHUNKING_CONFIG),
    )

    return PolicyRetriever(store=vs, config=retriever_config)


def _eval(chunks: RetrievedChunk, expected_doc_ids: list) -> tuple[int, float]:
    """Score the retrieved chunks of a single question.

    Args:
        chunks: Retrieved chunks, ordered best first.
        expected_doc_ids: Ids of the documents that answer the question.

    Returns:
        A ``(hit, reciprocal_rank)`` tuple: ``hit`` is 1 if an expected document
        appears in the chunks (else 0), and the reciprocal rank is ``1 / rank``
        of the first matching chunk (else 0.0).
    """
    hit: int = 0
    rr: float = 0
    rank = next((i for i, c in enumerate(chunks, 1) if c.chunk.doc_id in expected_doc_ids), None)
    hit, rr = (1, 1 / rank) if rank else (0, 0.0)

    return hit, rr


def run_eval(dataset: list[dict], retriever: PolicyRetriever) -> tuple[float, float, float]:
    """Evaluate a retriever on the ``retrieval_<eval_version>.json`` dataset.

    Questions with no expected document (out-of-corpus) are excluded from hit
    rate and MRR, and used only for the false-positive rate.

    Args:
        eval_version: Dataset version suffix (e.g. ``"v1"``).
        retriever: The retriever to evaluate.

    Returns:
        A ``(hit_rate, mrr, no_corpus_rate)`` tuple. ``no_corpus_rate`` is the
        share of out-of-corpus questions for which the retriever still returned
        chunks (lower is better).

    Raises:
        ValueError: If the dataset is empty.
    """

    n = len(dataset)
    if n == 0:
        raise ValueError("run_eval: dataset is empty")

    no_corpus_question = 0

    hit_k_tot = 0
    mrr_tot = 0
    no_corpus_tot = 0

    for item in dataset:
        chunks: list[RetrievedChunk] = retriever.retrieve(item["question"])
        if len(item["expected_doc_ids"]) == 0:
            no_corpus_question += 1

        hit_k, mrr = _eval(chunks, item["expected_doc_ids"])

        hit_k_tot += hit_k
        mrr_tot += mrr

        if len(item["expected_doc_ids"]) == 0 and len(chunks) > 0:
            no_corpus_tot += 1

    n = n - no_corpus_question
    hit_k_rate = 0 if n == 0 else hit_k_tot / n
    mrr_rate = 0 if n == 0 else mrr_tot / n
    no_corpus_rate = 0 if no_corpus_question == 0 else no_corpus_tot / no_corpus_question

    return hit_k_rate, mrr_rate, no_corpus_rate


def optimize_retriever_config(
    store: VectorStore,
    dataset: list[dict],
    ks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10),
    thresholds: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 21)),
) -> list[dict]:
    """Grid-search (k, max_distance) with a single vector search per question.

    Distances are Chroma distances (lower = closer), so a chunk is kept when
    distance <= threshold. Returns one row per (k, threshold) with
    hit@k, MRR (in-corpus questions) and false-positive rate (out-of-corpus).
    """
    max_k = max(ks)

    # 1) One search per question, raw distances, best first.
    cached: list[tuple[list[tuple[str, float]], set[str]]] = []
    for item in dataset:
        hits = store.similarity_search_with_score(query=item["question"], k=max_k)
        ranked = sorted(((doc.metadata["doc_id"], dist) for doc, dist in hits), key=lambda x: x[1])
        cached.append((ranked, set(item["expected_doc_ids"])))

    # 2) Offline sweep: no further embedding calls.
    rows: list[dict] = []
    for k in ks:
        for t in thresholds:
            hit = fp = n_in = n_out = 0
            rr = 0.0
            for ranked, expected in cached:
                kept = [doc_id for doc_id, dist in ranked[:k] if dist <= t]
                if expected:
                    n_in += 1
                    rank = next((i for i, d in enumerate(kept, start=1) if d in expected), None)
                    if rank is not None:
                        hit += 1
                        rr += 1 / rank
                else:
                    n_out += 1
                    fp += bool(kept)
            rows.append(
                {
                    "k": k,
                    "max_distance": t,
                    "hit@k": hit / n_in if n_in else 0.0,
                    "mrr": rr / n_in if n_in else 0.0,
                    "fp_rate": fp / n_out if n_out else 0.0,
                }
            )
    return rows


def main() -> int:
    config = RetrieverConfig(k=4, max_distance=0.4)
    hit_k, mrr, fp = run_eval(
        _load_dataset("retrieval_v1.json"),
        _get_retriever(settings=get_settings(), retriever_config=config),
    )
    print(
        f"k: {config.k}, max_distance: {config.max_distance}, hit@k: {hit_k}, mrr: {mrr}, fp: {fp}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

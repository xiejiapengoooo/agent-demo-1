import json
import logging
import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from functools import lru_cache
from threading import RLock
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from . import persisting
from .embedding import EMBEDDING_MODEL, embed_chunks

jieba.setLogLevel(logging.WARNING)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BM25_CANDIDATE_MULTIPLIER = 5
BM25_MIN_CANDIDATES = 20
RERANKER_MAX_LENGTH = 512

_RERANKER_LOCK = RLock()


def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    query = _validate_request(query, top_k, min_score)
    with persisting._locked_store():
        index, manifest = persisting.load_index()
        rows = _read_chunks()
    _validate_index(index, manifest, len(rows))
    if not rows or index.ntotal == 0:
        return []

    candidate_count = _candidate_count(len(rows), top_k)
    bm25_candidates = _bm25_candidates(rows, query, candidate_count)
    query_vector = _embed_query(query, index.d)
    vector_candidates = _faiss_candidates(
        index,
        rows,
        query_vector,
        candidate_count,
    )
    candidates = _merge_candidates(bm25_candidates, vector_candidates)
    scores = _rerank(query, [row["text"] for row in candidates])
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-item[1], int(item[0]["id"])),
    )
    return [
        {**row, "score": score}
        for row, score in ranked
        if min_score is None or score >= min_score
    ][:top_k]


def _validate_request(
    query: str,
    top_k: int,
    min_score: float | None,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if min_score is not None:
        if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
            raise TypeError("min_score must be a number")
        if not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1")
    return query.strip()


def _bm25_candidates(
    rows: Sequence[Mapping[str, Any]],
    query: str,
    candidate_count: int,
) -> list[Mapping[str, Any]]:
    corpus = [_tokenize(row["text"]) for row in rows]
    query_tokens = _tokenize(query) or [query.casefold()]
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    if len(scores) != len(rows) or any(
        not math.isfinite(float(score)) for score in scores
    ):
        raise ValueError("BM25 returned invalid candidate scores")

    indices = sorted(
        range(len(rows)),
        key=lambda index: (-float(scores[index]), int(rows[index]["id"])),
    )
    return [rows[index] for index in indices[:candidate_count]]


def _candidate_count(total: int, top_k: int) -> int:
    return min(total, max(top_k * BM25_CANDIDATE_MULTIPLIER, BM25_MIN_CANDIDATES))


def _faiss_candidates(
    index: Any,
    rows: Sequence[Mapping[str, Any]],
    query_vector: np.ndarray,
    candidate_count: int,
) -> list[Mapping[str, Any]]:
    scores, chunk_ids = index.search(query_vector, min(candidate_count, index.ntotal))
    rows_by_id = {int(row["id"]): row for row in rows}
    candidates = []
    for score, chunk_id in zip(scores[0], chunk_ids[0], strict=True):
        if chunk_id < 0:
            continue
        if not math.isfinite(float(score)):
            raise ValueError("FAISS returned a non-finite similarity score")
        row = rows_by_id.get(int(chunk_id))
        if row is None:
            raise ValueError(f"FAISS vector {chunk_id} has no matching chunk")
        candidates.append(row)
    return candidates


def _merge_candidates(
    *candidate_groups: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for group in candidate_groups:
        for row in group:
            merged[str(row["id"])] = row
    return list(merged.values())


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in jieba.lcut(text) if token.strip()]


def _validate_index(index: Any, manifest: Mapping[str, Any], chunk_count: int) -> None:
    import faiss

    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("query embedding model does not match the existing index")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index must use inner product for cosine similarity")
    if index.ntotal != chunk_count:
        raise ValueError("FAISS and SQLite chunk counts do not match")


def _embed_query(query: str, dimension: int) -> np.ndarray:
    import faiss

    embedded = embed_chunks([{"type": "text", "text": query}])
    if len(embedded) != 1:
        raise RuntimeError("query embedding returned an unexpected number of vectors")

    vector = np.asarray([embedded[0]["vector"]], dtype=np.float32)
    if vector.ndim != 2:
        raise ValueError("query embedding must be a two-dimensional vector")
    if vector.shape[1] != dimension:
        raise ValueError(
            f"query embedding dimension {vector.shape[1]} does not match "
            f"index dimension {dimension}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("query embedding must contain only finite values")
    if not np.linalg.norm(vector):
        raise ValueError("query embedding must have a non-zero norm")
    faiss.normalize_L2(vector)
    return vector


@lru_cache(maxsize=1)
def _load_reranker() -> tuple[Any, Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with _RERANKER_LOCK:
        device = _reranker_device()
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        model.to(device)
        model.eval()
        return tokenizer, model, device


def _reranker_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rerank(query: str, passages: Sequence[str]) -> list[float]:
    import torch

    tokenizer, model, device = _load_reranker()
    with _RERANKER_LOCK, torch.inference_mode():
        inputs = tokenizer(
            [query] * len(passages),
            list(passages),
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        logits = model(**inputs, return_dict=True).logits.reshape(-1)
        scores = torch.sigmoid(logits).detach().cpu().tolist()

    if len(scores) != len(passages) or any(
        not math.isfinite(float(score)) or not 0 <= float(score) <= 1
        for score in scores
    ):
        raise ValueError("reranker returned invalid scores")
    return [float(score) for score in scores]


def _read_chunks() -> list[dict[str, Any]]:
    database_path = (persisting.DATA_DIRECTORY / persisting.DATABASE_FILENAME).resolve()
    with (
        persisting._locked_store(),
        closing(
            sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
        ) as connection,
    ):
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT chunks.*, documents.file_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.id
            """
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def _chunk_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"])
    if not isinstance(metadata, dict):
        raise TypeError(f"chunk {row['id']} metadata must be a JSON object")
    return {
        **metadata,
        "id": str(row["id"]),
        "document_id": row["document_id"],
        "order": row["chunk_order"],
        "type": row["chunk_type"],
        "text": row["text"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "updated_at": row["updated_at"],
        "file_name": row["file_name"],
    }


__all__ = ["retrieve_chunks"]

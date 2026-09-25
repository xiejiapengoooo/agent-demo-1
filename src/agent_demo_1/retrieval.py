from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

import dashscope
import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .embedding import EMBEDDING_MODEL, _response_value, embed_chunks
from .persisting import DATA_DIRECTORY, DATABASE_FILENAME, _locked_store, load_index

RERANK_MODEL = "qwen3.7-text-rerank"
_CANDIDATE_MULTIPLIER = 5
_MIN_CANDIDATES = 20


def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if min_score is not None and (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not math.isfinite(float(min_score))
    ):
        raise ValueError("min_score must be a finite number or None")

    query = query.strip()

    with _locked_store():
        chunks = _read_chunks()
        if not chunks:
            return []
        index, manifest = load_index()
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("embedding model does not match the existing index")

    candidate_count = min(
        len(chunks),
        max(_MIN_CANDIDATES, top_k * _CANDIDATE_MULTIPLIER),
    )
    chunks_by_id = {int(chunk["id"]): chunk for chunk in chunks}

    candidate_ids: list[int] = []
    seen_ids: set[int] = set()

    query_vector = _embed_query(query)
    if query_vector.shape[0] != index.d:
        raise ValueError(
            f"query embedding dimension {query_vector.shape[0]} does not match "
            f"FAISS dimension {index.d}"
        )
    faiss.normalize_L2(query_vector.reshape(1, -1))
    _, labels = index.search(query_vector.reshape(1, -1), candidate_count)
    for label in labels[0]:
        chunk_id = int(label)
        if chunk_id >= 0 and chunk_id in chunks_by_id and chunk_id not in seen_ids:
            candidate_ids.append(chunk_id)
            seen_ids.add(chunk_id)

    bm25 = BM25Okapi([_tokenize(chunk["text"]) for chunk in chunks])
    bm25_scores = np.asarray(bm25.get_scores(_tokenize(query)))
    for position in np.argsort(bm25_scores)[::-1][:candidate_count]:
        chunk_id = int(chunks[position]["id"])
        if chunk_id not in seen_ids:
            candidate_ids.append(chunk_id)
            seen_ids.add(chunk_id)

    if not candidate_ids:
        return []

    candidates = [chunks_by_id[chunk_id] for chunk_id in candidate_ids]
    rerank_scores = _rerank(query, candidates)
    ranked = sorted(
        zip(candidates, rerank_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []
    for chunk, score in ranked:
        if min_score is not None and score < min_score:
            continue
        result = dict(chunk)
        result["score"] = score
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in jieba.lcut(text, cut_all=False) if token.strip()]


def _read_chunks() -> list[dict[str, Any]]:
    database_path = Path(DATA_DIRECTORY) / DATABASE_FILENAME
    if not database_path.is_file():
        return []

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                chunks.id,
                chunks.document_id,
                chunks.chunk_order,
                chunks.chunk_type,
                chunks.text,
                chunks.page_start,
                chunks.page_end,
                chunks.metadata_json,
                documents.file_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.id
            """
        ).fetchall()

    chunks = []
    for row in rows:
        chunk: dict[str, Any] = {
            "id": str(row["id"]),
            "document_id": row["document_id"],
            "order": row["chunk_order"],
            "type": row["chunk_type"],
            "text": row["text"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "file_name": row["file_name"],
        }
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, Mapping):
            for key, value in metadata.items():
                chunk.setdefault(str(key), value)
        chunks.append(chunk)
    return chunks


def _embed_query(query: str) -> np.ndarray:
    embedded = embed_chunks([{"text": query}])
    if len(embedded) != 1 or "vector" not in embedded[0]:
        raise RuntimeError("query embedding returned an invalid result")
    vector = np.asarray(embedded[0]["vector"], dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("query embedding must be a finite, non-empty vector")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("query embedding must have a finite, non-zero norm")
    return vector


def _rerank(query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return DashScope relevance scores in the original candidate order."""
    if not candidates:
        return []

    # Keep candidates in one request: scores are relative to that request.
    response = dashscope.TextReRank.call(
        model=RERANK_MODEL,
        query=query,
        documents=[candidate["text"] for candidate in candidates],
        top_n=len(candidates),
    )
    status_code = _response_value(response, "status_code")
    if status_code != HTTPStatus.OK:
        code = _response_value(response, "code", "unknown")
        message = _response_value(response, "message", "unknown error")
        raise RuntimeError(
            f"DashScope rerank failed with status {status_code}: {code} - {message}"
        )

    results = _response_value(_response_value(response, "output"), "results")
    if (
        not isinstance(results, Sequence)
        or isinstance(results, (str, bytes))
        or len(results) != len(candidates)
    ):
        raise RuntimeError("DashScope rerank returned incomplete results")

    scores: list[float | None] = [None] * len(candidates)
    for result in results:
        index = _response_value(result, "index")
        score = _response_value(result, "relevance_score")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(candidates)
            or scores[index] is not None
        ):
            raise RuntimeError(
                "DashScope rerank returned an invalid or duplicate index"
            )
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise RuntimeError("DashScope rerank returned an invalid relevance score")
        scores[index] = float(score)
    return [score for score in scores if score is not None]


__all__ = ["retrieve_chunks"]

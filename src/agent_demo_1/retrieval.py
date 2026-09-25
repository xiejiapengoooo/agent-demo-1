import json
import logging
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from functools import lru_cache
from threading import RLock
from typing import Any

import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import persisting

jieba.setLogLevel(logging.WARNING)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BM25_CANDIDATE_MULTIPLIER = 5
BM25_MIN_CANDIDATES = 20

_RERANKER_LOCK = RLock()


def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
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

    rows = _read_chunks()
    if not rows:
        return []

    query = query.strip()
    candidate_indices = _bm25_candidates(rows, query, top_k)
    if not candidate_indices:
        return []

    passages = [rows[index]["text"] for index in candidate_indices]
    reranked = _rerank(query, passages)
    if len(reranked) != len(candidate_indices):
        raise RuntimeError(
            f"reranker returned {len(reranked)} scores; "
            f"expected {len(candidate_indices)}"
        )

    matches = sorted(
        (
            (int(rows[index]["id"]), _score_to_float(score))
            for index, score in zip(candidate_indices, reranked)
        ),
        key=lambda match: (-match[1], match[0]),
    )
    if min_score is not None:
        matches = [match for match in matches if match[1] >= min_score]
    return _read_matches(matches[:top_k])


def _bm25_candidates(
    rows: Sequence[sqlite3.Row],
    query: str,
    top_k: int,
) -> list[int]:
    tokenized_corpus = [_tokenize(row["text"]) for row in rows]
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        tokenized_query = [query.casefold()]

    scores = np.asarray(BM25Okapi(tokenized_corpus).get_scores(tokenized_query))
    if scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
        raise ValueError("BM25 returned invalid candidate scores")

    candidate_count = min(
        len(rows),
        max(top_k * BM25_CANDIDATE_MULTIPLIER, BM25_MIN_CANDIDATES),
    )
    return sorted(
        range(len(rows)),
        key=lambda index: (-float(scores[index]), int(rows[index]["id"])),
    )[:candidate_count]


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in jieba.lcut(text) if token.strip()]


@lru_cache(maxsize=1)
def _load_reranker() -> tuple[Any, Any, torch.device]:
    with _RERANKER_LOCK:
        device = _reranker_device()
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        model.to(device)
        model.eval()
        return tokenizer, model, device


def _reranker_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rerank(query: str, passages: Sequence[str]) -> list[float]:
    tokenizer, model, device = _load_reranker()
    with _RERANKER_LOCK, torch.inference_mode():
        inputs = tokenizer(
            [query] * len(passages),
            list(passages),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {name: value.to(device) for name, value in inputs.items()}
        output = model(**model_inputs, return_dict=True)
        logits = output.logits.reshape(-1)
        scores = torch.sigmoid(logits).detach().cpu().tolist()

    if len(scores) != len(passages) or not all(
        math.isfinite(float(score)) for score in scores
    ):
        raise ValueError("reranker returned invalid scores")
    return [float(score) for score in scores]


def _score_to_float(score: float) -> float:
    value = float(score)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reranker returned a score outside [0, 1]")
    return value


def _read_chunks() -> list[sqlite3.Row]:
    database_path = (persisting.DATA_DIRECTORY / persisting.DATABASE_FILENAME).resolve()
    with (
        persisting._locked_store(),
        closing(
            sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
        ) as connection,
    ):
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT chunks.*, documents.file_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.id
            """
        ).fetchall()


def _read_matches(matches: Sequence[tuple[int, float]]) -> list[dict[str, Any]]:
    if not matches:
        return []

    database_path = (persisting.DATA_DIRECTORY / persisting.DATABASE_FILENAME).resolve()
    rows_by_id: dict[int, sqlite3.Row] = {}
    with (
        persisting._locked_store(),
        closing(
            sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
        ) as connection,
    ):
        connection.row_factory = sqlite3.Row
        batch_size = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        for start in range(0, len(matches), batch_size):
            chunk_ids = [
                chunk_id for chunk_id, _ in matches[start : start + batch_size]
            ]
            placeholders = ", ".join("?" for _ in chunk_ids)
            rows = connection.execute(
                f"""
                SELECT chunks.*, documents.file_name
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
            rows_by_id.update((row["id"], row) for row in rows)

    results = []
    for chunk_id, score in matches:
        row = rows_by_id.get(chunk_id)
        if row is None:
            raise ValueError(
                f"retrieved chunk {chunk_id} has no matching chunk and document"
            )
        metadata = json.loads(row["metadata_json"])
        if not isinstance(metadata, dict):
            raise TypeError(f"chunk {row['id']} metadata must be a JSON object")
        results.append(
            {
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
                "score": score,
            }
        )
    return results


__all__ = ["retrieve_chunks"]

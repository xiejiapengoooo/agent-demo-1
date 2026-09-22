import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from typing import Any

import faiss
import numpy as np

from . import persisting
from .embedding import EMBEDDING_MODEL, embed_chunks


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

    index, manifest = persisting.load_index()
    _validate_index(index, manifest)
    if index.ntotal == 0:
        return []

    # Keep the network request outside the store lock so uploads can continue.
    query_vector = _embed_query(query.strip())

    with persisting._locked_store():
        if query_vector.shape[1] != index.d:
            raise ValueError(
                f"query embedding dimension {query_vector.shape[1]} does not match "
                f"index dimension {index.d}"
            )

        scores, chunk_ids = index.search(query_vector, min(top_k, index.ntotal))
        matches = []
        for score, chunk_id in zip(scores[0], chunk_ids[0]):
            if chunk_id < 0:
                continue
            if not math.isfinite(float(score)):
                raise ValueError("FAISS returned a non-finite similarity score")
            similarity = float(np.clip(score, -1.0, 1.0))
            if min_score is None or similarity >= min_score:
                matches.append((int(chunk_id), similarity))

        return _read_matches(matches)


def _validate_index(index: Any, manifest: dict[str, Any]) -> None:
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("query embedding model does not match the existing index")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index must use inner product for cosine similarity")


def _embed_query(query: str) -> np.ndarray:
    chunks = embed_chunks([{"type": "text", "text": query}])
    vector = np.asarray([chunks[0]["vector"]], dtype=np.float32)
    if vector.ndim != 2 or vector.shape[1] == 0:
        raise ValueError("query embedding must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("query embedding must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("query embedding must have a finite, non-zero norm")
    faiss.normalize_L2(vector)
    return vector


def _read_matches(matches: Sequence[tuple[int, float]]) -> list[dict[str, Any]]:
    if not matches:
        return []

    database_path = (persisting.DATA_DIRECTORY / persisting.DATABASE_FILENAME).resolve()
    rows_by_id: dict[int, sqlite3.Row] = {}
    with closing(
        sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    ) as connection:
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
                f"FAISS vector {chunk_id} has no matching chunk and document"
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

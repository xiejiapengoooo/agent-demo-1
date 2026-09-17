import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from .embedding import embed_chunks

DATABASE_PATH = Path("data/index.db")


def _chunk_texts(chunks: Sequence[Mapping[str, Any]]) -> list[str]:
    texts = []
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"chunk at index {index} has no chunk_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"chunk {chunk_id!r} has no document_id")

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"chunk {chunk_id!r} must have non-empty text")
        texts.append(text)
    return texts


def persist_chunks(
    chunks: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    database_path: str | Path = DATABASE_PATH,
    embedding_model: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("each chunk must have exactly one embedding")

    texts = _chunk_texts(chunks)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")

        rows.append(
            (
                chunk_id,
                document_id,
                chunk.get("order"),
                chunk.get("type"),
                texts[index],
                chunk.get("page_start"),
                chunk.get("page_end"),
                json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                json.dumps(embedding, separators=(",", ":")),
                embedding_model,
            )
        )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_order INTEGER,
                chunk_type TEXT,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                chunk_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                PRIMARY KEY (document_id, chunk_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_order
            ON chunks (document_id, chunk_order)
            """
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                document_id,
                chunk_order,
                chunk_type,
                text,
                page_start,
                page_end,
                chunk_json,
                embedding_json,
                embedding_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (document_id, chunk_id) DO UPDATE SET
                chunk_order = excluded.chunk_order,
                chunk_type = excluded.chunk_type,
                text = excluded.text,
                page_start = excluded.page_start,
                page_end = excluded.page_end,
                chunk_json = excluded.chunk_json,
                embedding_json = excluded.embedding_json,
                embedding_model = excluded.embedding_model
            """,
            rows,
        )


def build_index():
    with Path("mock.json").open(encoding="utf-8") as data_file:
        chunks = json.load(data_file)

    embeddings = embed_chunks(chunks)
    # persist_chunks(
    #     chunks,
    #     embeddings,
    #     database_path=database_path,
    #     embedding_model=embedding_model,
    # )
    # return len(chunks)


__all__ = ["build_index", "persist_chunks"]

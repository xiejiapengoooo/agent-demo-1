import base64
import binascii
import hashlib
import json
import math
import mimetypes
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, unquote_to_bytes, urlparse
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

import faiss
import numpy as np

DATA_DIRECTORY = Path("data")
FAISS_FILENAME = "index.faiss"
DATABASE_FILENAME = "index.sqlite3"
MANIFEST_FILENAME = "manifest.json"
IMAGES_DIRECTORY = "images"
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30

SCHEMA_VERSION = 2
CORE_FIELDS = frozenset(
    {
        "vector_id",
        "chunk_id",
        "document_id",
        "order",
        "type",
        "text",
        "page_start",
        "page_end",
        "vector",
    }
)

IMAGE_EXTENSIONS = {
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class _ImageAsset:
    relative_path: str
    content: bytes


def persist_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    embedding_model: str,
) -> dict[str, Any]:
    """Persist a complete chunk snapshot to SQLite and FAISS."""
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string")

    rows, vectors, image_assets = _prepare_chunks(chunks)

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()
    build_id = str(uuid4())
    temporary_directory = Path(tempfile.mkdtemp(prefix=".temp-", dir=DATA_DIRECTORY))

    try:
        temporary_index = temporary_directory / FAISS_FILENAME
        temporary_database = temporary_directory / DATABASE_FILENAME
        temporary_manifest = temporary_directory / MANIFEST_FILENAME
        temporary_images = temporary_directory / IMAGES_DIRECTORY

        index = _build_faiss_index(vectors)
        faiss.write_index(index, str(temporary_index))
        index_sha256 = _file_sha256(temporary_index)

        _write_database(
            temporary_database,
            rows,
            embedding_model=embedding_model.strip(),
            embedding_dimension=int(vectors.shape[1]),
            created_at=created_at,
            build_id=build_id,
            index_sha256=index_sha256,
        )
        _write_images(temporary_images, image_assets)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": created_at,
            "embedding_model": embedding_model.strip(),
            "embedding_dimension": int(vectors.shape[1]),
            "metric": "cosine",
            "normalization": "l2",
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "chunk_count": len(rows),
            "image_count": len(image_assets),
            "index_sha256": index_sha256,
            "files": {
                "faiss": FAISS_FILENAME,
                "sqlite": DATABASE_FILENAME,
                "images": IMAGES_DIRECTORY,
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (DATA_DIRECTORY / IMAGES_DIRECTORY).mkdir(parents=True, exist_ok=True)
        for image in image_assets:
            source = temporary_directory / image.relative_path
            destination = DATA_DIRECTORY / image.relative_path
            os.replace(source, destination)

        # The manifest is replaced last and acts as the snapshot commit marker.
        os.replace(temporary_index, DATA_DIRECTORY / FAISS_FILENAME)
        os.replace(temporary_database, DATA_DIRECTORY / DATABASE_FILENAME)
        os.replace(temporary_manifest, DATA_DIRECTORY / MANIFEST_FILENAME)
        return manifest
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def load_index(
    data_directory: str | Path = DATA_DIRECTORY,
) -> tuple[Any, dict[str, Any]]:
    """Load a persisted index after verifying its SQLite mapping and checksum."""
    data_directory = Path(data_directory)
    index_path = data_directory / FAISS_FILENAME
    database_path = data_directory / DATABASE_FILENAME
    manifest_path = data_directory / MANIFEST_FILENAME

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing index manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid index manifest: {manifest_path}") from error

    expected_checksum = manifest.get("index_sha256")
    if not isinstance(expected_checksum, str):
        raise TypeError("manifest has no index_sha256")
    if _file_sha256(index_path) != expected_checksum:
        raise ValueError("FAISS index checksum does not match the manifest")

    with sqlite3.connect(database_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM store_metadata"))
        mapping_count = connection.execute(
            "SELECT COUNT(*) FROM faiss_mapping"
        ).fetchone()[0]
        image_metadata = connection.execute(
            "SELECT metadata_json FROM chunks WHERE chunk_type = 'image'"
        ).fetchall()

    if metadata.get("build_id") != manifest.get("build_id"):
        raise ValueError("SQLite database and manifest belong to different snapshots")
    if metadata.get("index_sha256") != expected_checksum:
        raise ValueError(
            "SQLite database and FAISS index belong to different snapshots"
        )
    if mapping_count != manifest.get("chunk_count"):
        raise ValueError("SQLite FAISS mapping count does not match the manifest")
    if len(image_metadata) != manifest.get("image_count"):
        raise ValueError("SQLite image count does not match the manifest")
    _verify_images(data_directory, image_metadata)

    index = faiss.read_index(str(index_path))
    if index.ntotal != manifest.get("chunk_count"):
        raise ValueError("FAISS vector count does not match the manifest")
    if index.d != manifest.get("embedding_dimension"):
        raise ValueError("FAISS dimension does not match the manifest")
    return index, manifest


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Any, ...]], np.ndarray, list[_ImageAsset]]:
    if not chunks:
        raise ValueError("at least one chunk is required")

    rows: list[tuple[Any, ...]] = []
    vectors: list[list[float]] = []
    image_assets: list[_ImageAsset] = []
    seen_chunks: set[tuple[str, str]] = set()
    seen_vector_ids: set[str] = set()
    expected_dimension: int | None = None
    updated_at = datetime.now(UTC).isoformat()

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"chunk at index {index} must be a mapping")

        chunk_id = _required_uuid(chunk, "chunk_id", index)
        document_id = _required_string(chunk, "document_id", index)
        identity = (document_id, chunk_id)
        if identity in seen_chunks:
            raise ValueError(
                f"duplicate chunk_id {chunk_id!r} in document {document_id!r}"
            )
        seen_chunks.add(identity)

        vector_id = _required_uuid(chunk, "vector_id", index)
        if vector_id in seen_vector_ids:
            raise ValueError(f"duplicate vector_id {vector_id!r}")
        seen_vector_ids.add(vector_id)
        chunk_order = _required_integer(chunk, "order", index)
        chunk_type = _required_string(chunk, "type", index)
        text = _required_string(chunk, "text", index)
        page_start = _optional_integer(chunk, "page_start", index)
        page_end = _optional_integer(chunk, "page_end", index)
        if page_start is not None and page_end is not None and page_start > page_end:
            raise ValueError(f"chunk at index {index} has page_start after page_end")

        vector = _vector(chunk, index)
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError(
                f"chunk at index {index} has dimension {len(vector)}; "
                f"expected {expected_dimension}"
            )
        vectors.append(vector)

        metadata = {
            key: value for key, value in chunk.items() if key not in CORE_FIELDS
        }
        if chunk_type == "image":
            image_asset, mime_type, image_sha256 = _prepare_image(
                chunk.get("source"), index
            )
            image_assets.append(image_asset)
            metadata["source"] = image_asset.relative_path
            metadata["source_mime_type"] = mime_type
            metadata["source_sha256"] = image_sha256
        rows.append(
            (
                vector_id,
                chunk_id,
                document_id,
                chunk_order,
                chunk_type,
                text,
                page_start,
                page_end,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                updated_at,
            )
        )

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must not be zero vectors")
    return rows, matrix, image_assets


def _prepare_image(
    source: Any,
    chunk_index: int,
) -> tuple[_ImageAsset, str, str]:
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"image chunk at index {chunk_index} has no source")

    value = source.strip()
    if value.startswith("data:"):
        mime_type, content = _decode_image_data_uri(value, chunk_index)
    else:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            if not parsed.netloc:
                raise ValueError(
                    f"image chunk at index {chunk_index} has an invalid remote URL"
                )
            mime_type, content = _download_image(value, chunk_index)
        else:
            image_path = (
                Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(value)
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"image chunk at index {chunk_index} source does not exist: "
                    f"{image_path}"
                )
            mime_type, _ = mimetypes.guess_type(image_path.name)
            if mime_type is None or not mime_type.startswith("image/"):
                raise ValueError(
                    f"image chunk at index {chunk_index} has an unsupported source type"
                )
            content = image_path.read_bytes()

    if not content:
        raise ValueError(f"image chunk at index {chunk_index} has an empty source")

    image_sha256 = hashlib.sha256(content).hexdigest()
    extension = IMAGE_EXTENSIONS.get(mime_type)
    if extension is None:
        extension = mimetypes.guess_extension(mime_type)
    if extension is None:
        raise ValueError(
            f"image chunk at index {chunk_index} has unsupported "
            f"MIME type {mime_type!r}"
        )
    if extension == ".jpe":
        extension = ".jpg"

    filename = f"{image_sha256[:16]}{extension}"
    relative_path = (Path(IMAGES_DIRECTORY) / filename).as_posix()
    return _ImageAsset(relative_path, content), mime_type, image_sha256


def _decode_image_data_uri(value: str, chunk_index: int) -> tuple[str, bytes]:
    header, separator, payload = value.partition(",")
    if not separator:
        raise ValueError(f"image chunk at index {chunk_index} has an invalid data URI")

    parts = header[5:].split(";")
    mime_type = parts[0].lower()
    if not mime_type.startswith("image/"):
        raise ValueError(f"image chunk at index {chunk_index} has a non-image data URI")

    try:
        if "base64" in {part.lower() for part in parts[1:]}:
            content = base64.b64decode("".join(payload.split()), validate=True)
        else:
            content = unquote_to_bytes(payload)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ValueError(
            f"image chunk at index {chunk_index} has invalid image data"
        ) from error
    return mime_type, content


def _download_image(url: str, chunk_index: int) -> tuple[str, bytes]:
    request = Request(url, headers={"User-Agent": "agent-demo-1/0.1"})
    try:
        response = urlopen(request, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
    except HTTPError as error:
        raise ValueError(
            f"image chunk at index {chunk_index} download failed with HTTP {error.code}"
        ) from error
    except (OSError, TimeoutError, URLError, ValueError) as error:
        raise ValueError(
            f"image chunk at index {chunk_index} download failed: {error}"
        ) from error

    with response:
        final_url = response.geturl()
        if urlparse(final_url).scheme not in {"http", "https"}:
            raise ValueError(
                f"image chunk at index {chunk_index} redirected to an unsupported URL"
            )

        content_type = response.headers.get("Content-Type")
        mime_type = (
            content_type.split(";", 1)[0].strip().lower() if content_type else ""
        )
        if mime_type in {"", "application/octet-stream"}:
            mime_type, _ = mimetypes.guess_type(urlparse(final_url).path)
        if mime_type is None or not mime_type.startswith("image/"):
            raise ValueError(
                f"image chunk at index {chunk_index} download returned "
                f"non-image content type {content_type!r}"
            )

        try:
            content = response.read()
        except (OSError, TimeoutError) as error:
            raise ValueError(
                f"image chunk at index {chunk_index} download failed: {error}"
            ) from error
        return mime_type, content


def _write_images(directory: Path, image_assets: Sequence[_ImageAsset]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for image in image_assets:
        (directory / Path(image.relative_path).name).write_bytes(image.content)


def _verify_images(
    data_directory: Path,
    image_metadata: Sequence[tuple[str]],
) -> None:
    for row in image_metadata:
        try:
            metadata = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("image chunk has invalid metadata_json") from error

        source = metadata.get("source")
        expected_checksum = metadata.get("source_sha256")
        if not isinstance(source, str) or not isinstance(expected_checksum, str):
            raise ValueError("image chunk metadata has no persisted source")

        relative_path = Path(source)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != IMAGES_DIRECTORY
        ):
            raise ValueError(f"image chunk has an invalid source path: {source!r}")

        image_path = data_directory / relative_path
        if _file_sha256(image_path) != expected_checksum:
            raise ValueError(f"persisted image checksum does not match: {source}")


def _build_faiss_index(vectors: np.ndarray) -> Any:
    normalized_vectors = vectors.copy()
    faiss.normalize_L2(normalized_vectors)

    dimension = int(normalized_vectors.shape[1])
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    faiss_ids = np.arange(len(normalized_vectors), dtype=np.int64)
    index.add_with_ids(normalized_vectors, faiss_ids)
    return index


def _write_database(
    path: Path,
    rows: Sequence[tuple[Any, ...]],
    *,
    embedding_model: str,
    embedding_dimension: int,
    created_at: str,
    build_id: str,
    index_sha256: str,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE chunks (
                vector_id UUID PRIMARY KEY,
                chunk_id UUID NOT NULL,
                document_id STR NOT NULL,
                chunk_order INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (document_id, chunk_id)
            );

            CREATE INDEX idx_chunks_document_order
            ON chunks (document_id, chunk_order);

            CREATE TABLE faiss_mapping (
                faiss_id INTEGER PRIMARY KEY,
                vector_id UUID NOT NULL UNIQUE,
                FOREIGN KEY (vector_id) REFERENCES chunks (vector_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                vector_id,
                chunk_id,
                document_id,
                chunk_order,
                chunk_type,
                text,
                page_start,
                page_end,
                metadata_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.executemany(
            "INSERT INTO faiss_mapping (faiss_id, vector_id) VALUES (?, ?)",
            ((faiss_id, row[0]) for faiss_id, row in enumerate(rows)),
        )
        connection.executemany(
            "INSERT INTO store_metadata (key, value) VALUES (?, ?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("build_id", build_id),
                ("created_at", created_at),
                ("embedding_model", embedding_model),
                ("embedding_dimension", str(embedding_dimension)),
                ("metric", "cosine"),
                ("index_sha256", index_sha256),
            ),
        )


def _required_uuid(chunk: Mapping[str, Any], key: str, index: int) -> str:
    value = _required_string(chunk, key, index)
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError(f"chunk at index {index} has an invalid {key}") from error


def _required_string(chunk: Mapping[str, Any], key: str, index: int) -> str:
    value = chunk.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"chunk at index {index} has no {key}")
    return value.strip()


def _required_integer(chunk: Mapping[str, Any], key: str, index: int) -> int:
    value = chunk.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"chunk at index {index} has an invalid {key}")
    return value


def _optional_integer(chunk: Mapping[str, Any], key: str, index: int) -> int | None:
    value = chunk.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"chunk at index {index} has an invalid {key}")
    return value


def _vector(chunk: Mapping[str, Any], index: int) -> list[float]:
    values = chunk.get("vector")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"chunk at index {index} has no vector")
    if not values:
        raise ValueError(f"chunk at index {index} has an empty vector")

    vector = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"chunk at index {index} has a non-numeric vector value")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"chunk at index {index} has a non-finite vector value")
        vector.append(numeric_value)
    return vector


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["load_index", "persist_chunks"]

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
DATABASE_FILENAME = "index.sqlite3"
FAISS_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.json"
IMAGES_DIRECTORY = "images"

PENDING_STATUS = "pending"
COMPLETED_STATUS = "completed"
SCHEMA_VERSION = "1.0.0"
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30

CORE_FIELDS = frozenset(
    {
        "id",
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


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    file_sha256: str
    file_name: str
    updated_at: str
    status: str


@dataclass(frozen=True, slots=True)
class ImageAsset:
    relative_path: str
    content: bytes


def initialize_database() -> Path:
    data_directory = Path(DATA_DIRECTORY)
    data_directory.mkdir(parents=True, exist_ok=True)
    database_path = data_directory / DATABASE_FILENAME

    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID NOT NULL PRIMARY KEY,
                file_sha256 TEXT NOT NULL,
                file_name TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id UUID NOT NULL PRIMARY KEY,
                vector_id TEXT NOT NULL,
                document_id UUID NOT NULL,
                chunk_order INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    return database_path


def read_pending_documents() -> list[Document]:
    database_path = initialize_database()
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, file_sha256, file_name, updated_at, status
            FROM documents
            WHERE status = ?
            ORDER BY updated_at, id
            """,
            (PENDING_STATUS,),
        ).fetchall()

    return [_document_from_row(row, index) for index, row in enumerate(rows)]


def database_chunk_count() -> int:
    database_path = initialize_database()
    with sqlite3.connect(database_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def persist_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    documents: Sequence[Document],
    embedding_model: str,
) -> dict[str, Any]:
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string")
    if not documents:
        raise ValueError("at least one document is required")

    database_path = initialize_database()
    existing_chunk_count = database_chunk_count()
    existing_manifest: dict[str, Any] | None = None
    index_exists = (DATA_DIRECTORY / FAISS_FILENAME).is_file()
    manifest_exists = (DATA_DIRECTORY / MANIFEST_FILENAME).is_file()
    if index_exists != manifest_exists:
        raise ValueError(
            "FAISS index and manifest must either both exist or both be absent"
        )

    if index_exists:
        existing_index, existing_manifest = load_index()
        if existing_index.ntotal != existing_chunk_count:
            raise ValueError("FAISS and SQLite chunk counts do not match")
        if existing_manifest.get("embedding_model") != embedding_model.strip():
            raise ValueError("embedding model does not match the existing index")
        base_index = faiss.clone_index(existing_index)
        next_vector_id = _manifest_next_vector_id(
            existing_manifest,
            database_path,
        )
    else:
        if existing_chunk_count:
            raise ValueError("chunks exist without a FAISS index and manifest")
        base_index = None
        next_vector_id = 0

    if len(chunks) > int(np.iinfo(np.int64).max) - next_vector_id + 1:
        raise OverflowError("no FAISS vector ids remain in the int64 range")

    rows, vectors, image_assets = _prepare_chunks(
        chunks,
        documents,
        vector_id_start=next_vector_id,
    )
    faiss_ids = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    created_at = _utc_now()
    temporary_directory = Path(tempfile.mkdtemp(prefix=".temp-", dir=DATA_DIRECTORY))

    try:
        temporary_index = temporary_directory / FAISS_FILENAME
        temporary_database = temporary_directory / DATABASE_FILENAME
        temporary_manifest = temporary_directory / MANIFEST_FILENAME
        temporary_images = temporary_directory / IMAGES_DIRECTORY

        if base_index is None:
            index = _build_faiss_index(vectors, faiss_ids)
        else:
            if int(vectors.shape[1]) != base_index.d:
                raise ValueError(
                    f"embedding dimension {vectors.shape[1]} does not match "
                    f"existing index dimension {base_index.d}"
                )
            normalized_vectors = vectors.copy()
            faiss.normalize_L2(normalized_vectors)
            base_index.add_with_ids(normalized_vectors, faiss_ids)
            index = base_index
        faiss.write_index(index, str(temporary_index))
        index_sha256 = file_sha256(temporary_index)

        _write_database_snapshot(
            database_path,
            temporary_database,
            rows,
            documents,
            completed_at=created_at,
        )
        (
            document_count,
            chunk_count,
        ) = _database_snapshot_counts(temporary_database)
        if chunk_count != index.ntotal:
            raise ValueError("FAISS and SQLite chunk counts do not match")
        _write_images(temporary_images, image_assets)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "embedding_model": embedding_model.strip(),
            "embedding_dimension": int(vectors.shape[1]),
            "document_count": document_count,
            "chunk_count": chunk_count,
            "next_vector_id": next_vector_id + len(rows),
            "index_sha256": index_sha256,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (DATA_DIRECTORY / IMAGES_DIRECTORY).mkdir(parents=True, exist_ok=True)
        for image in image_assets:
            source = temporary_images / Path(image.relative_path).name
            destination = DATA_DIRECTORY / image.relative_path
            os.replace(source, destination)

        # The manifest is published last and is the snapshot commit marker.
        os.replace(temporary_index, DATA_DIRECTORY / FAISS_FILENAME)
        os.replace(temporary_database, database_path)
        os.replace(temporary_manifest, DATA_DIRECTORY / MANIFEST_FILENAME)
        return manifest
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def load_index() -> tuple[Any, dict[str, Any]]:
    index_path = DATA_DIRECTORY / FAISS_FILENAME
    database_path = DATA_DIRECTORY / DATABASE_FILENAME
    manifest_path = DATA_DIRECTORY / MANIFEST_FILENAME

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing index manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid index manifest: {manifest_path}") from error

    expectedIndexSha256 = manifest.get("index_sha256")
    if not isinstance(expectedIndexSha256, str):
        raise TypeError("manifest has no 'index_sha256'")
    if file_sha256(index_path) != expectedIndexSha256:
        raise ValueError(f"{index_path.name} checksum does not match the manifest")

    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexIDMap2):
        raise TypeError("FAISS index must be IndexIDMap2")
    if index.ntotal != manifest.get("chunk_count"):
        raise ValueError("FAISS vector count does not match the manifest")
    if index.d != manifest.get("embedding_dimension"):
        raise ValueError("FAISS dimension does not match the manifest")

    faiss_ids = sorted(int(value) for value in faiss.vector_to_array(index.id_map))
    database_ids = sorted(_database_vector_ids(database_path))
    if faiss_ids != database_ids:
        raise ValueError("FAISS ids do not match chunks.vector_id values")
    _manifest_next_vector_id(manifest, database_path)
    return index, manifest


def delete_vectors(vector_ids: Sequence[int | str]) -> dict[str, Any]:
    ids = _normalize_vector_ids(vector_ids)
    if not ids:
        raise ValueError("at least one vector_id is required")

    database_path = initialize_database()
    existing_index, existing_manifest = load_index()
    existing_ids = set(_faiss_vector_ids(existing_index))
    missing_ids = sorted(set(ids) - existing_ids)
    if missing_ids:
        joined = ", ".join(str(value) for value in missing_ids)
        raise ValueError(f"vector ids do not exist: {joined}")

    index = faiss.clone_index(existing_index)
    ids_array = np.asarray(ids, dtype=np.int64)
    removed_count = int(index.remove_ids(ids_array))
    if removed_count != len(ids):
        raise RuntimeError(
            f"FAISS removed {removed_count} vectors; expected {len(ids)}"
        )

    created_at = _utc_now()
    temporary_directory = Path(tempfile.mkdtemp(prefix=".temp-", dir=DATA_DIRECTORY))
    try:
        temporary_index = temporary_directory / FAISS_FILENAME
        temporary_database = temporary_directory / DATABASE_FILENAME
        temporary_manifest = temporary_directory / MANIFEST_FILENAME

        faiss.write_index(index, str(temporary_index))
        index_sha256 = file_sha256(temporary_index)
        _copy_database(database_path, temporary_database)
        deleted_count = _delete_database_vectors(temporary_database, ids)
        if deleted_count != len(ids):
            raise RuntimeError(
                f"SQLite deleted {deleted_count} chunks; expected {len(ids)}"
            )

        document_count, chunk_count, image_count, image_asset_count = (
            _database_snapshot_counts(temporary_database)
        )
        if chunk_count != index.ntotal:
            raise ValueError("FAISS and SQLite chunk counts do not match")

        manifest = dict(existing_manifest)
        manifest.update(
            {
                "build_id": str(uuid4()),
                "created_at": created_at,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "image_count": image_count,
                "image_asset_count": image_asset_count,
                "index_sha256": index_sha256,
            }
        )
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary_index, DATA_DIRECTORY / FAISS_FILENAME)
        os.replace(temporary_database, database_path)
        os.replace(temporary_manifest, DATA_DIRECTORY / MANIFEST_FILENAME)
        return manifest
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
    documents: Sequence[Document],
    *,
    vector_id_start: int = 0,
) -> tuple[list[tuple[Any, ...]], np.ndarray, list[ImageAsset]]:
    if not chunks:
        raise ValueError("at least one chunk is required")

    document_ids = {document.id for document in documents}
    if len(document_ids) != len(documents):
        raise ValueError("documents contain duplicate ids")

    rows: list[tuple[Any, ...]] = []
    vectors: list[list[float]] = []
    image_assets: dict[str, ImageAsset] = {}
    seen_chunk_ids: set[str] = set()
    seen_orders: set[tuple[str, int]] = set()
    chunk_document_ids: set[str] = set()
    expected_dimension: int | None = None
    updated_at = _utc_now()

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"chunk at index {index} must be a mapping")

        chunk_id = str(uuid4())
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"duplicate chunk id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        document_id = _required_uuid(chunk, ("document_id",), index)
        if document_id not in document_ids:
            raise ValueError(
                f"chunk at index {index} references unknown document {document_id}"
            )
        chunk_document_ids.add(document_id)

        chunk_order = _required_integer(chunk, "order", index)
        if chunk_order < 0:
            raise ValueError(f"chunk at index {index} has a negative order")
        order_key = (document_id, chunk_order)
        if order_key in seen_orders:
            raise ValueError(
                f"duplicate chunk order {chunk_order} for document {document_id}"
            )
        seen_orders.add(order_key)

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
            image_assets[image_asset.relative_path] = image_asset
            metadata["source"] = image_asset.relative_path
            metadata["source_mime_type"] = mime_type
            metadata["source_sha256"] = image_sha256

        try:
            metadata_json = json.dumps(
                metadata, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"chunk at index {index} contains non-JSON metadata"
            ) from error

        rows.append(
            (
                chunk_id,
                str(vector_id_start + index),
                document_id,
                chunk_order,
                chunk_type,
                text,
                page_start,
                page_end,
                metadata_json,
                updated_at,
            )
        )

    missing_documents = document_ids - chunk_document_ids
    if missing_documents:
        joined = ", ".join(sorted(missing_documents))
        raise ValueError(f"documents produced no chunks: {joined}")

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must not be zero vectors")
    return rows, matrix, list(image_assets.values())


def _build_faiss_index(vectors: np.ndarray, faiss_ids: np.ndarray) -> Any:
    normalized_vectors = vectors.copy()
    faiss.normalize_L2(normalized_vectors)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(normalized_vectors.shape[1])))
    index.add_with_ids(normalized_vectors, faiss_ids)
    return index


def _write_database_snapshot(
    source: Path,
    destination: Path,
    rows: Sequence[tuple[Any, ...]],
    documents: Sequence[Document],
    *,
    completed_at: str,
) -> None:
    _copy_database(source, destination)

    with sqlite3.connect(destination) as connection:
        statuses = dict(connection.execute("SELECT id, status FROM documents"))
        for document in documents:
            if statuses.get(document.id) != PENDING_STATUS:
                raise RuntimeError(
                    f"document {document.id} is no longer in pending status"
                )
            existing_chunk = connection.execute(
                "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1",
                (document.id,),
            ).fetchone()
            if existing_chunk is not None:
                raise RuntimeError(f"document {document.id} already has chunks")

        connection.executemany(
            """
            INSERT INTO chunks (
                id,
                vector_id,
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
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            ((COMPLETED_STATUS, completed_at, document.id) for document in documents),
        )


def _database_snapshot_counts(path: Path) -> tuple[int, int]:
    with sqlite3.connect(path) as connection:
        document_count = connection.execute(
            "SELECT COUNT(DISTINCT document_id) FROM chunks"
        ).fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    return (
        int(document_count),
        int(chunk_count),
    )


def _manifest_next_vector_id(
    manifest: Mapping[str, Any],
    database_path: Path,
) -> int:
    next_vector_id = manifest.get("next_vector_id")
    if (
        not isinstance(next_vector_id, int)
        or isinstance(next_vector_id, bool)
        or next_vector_id < 0
        or next_vector_id > int(np.iinfo(np.int64).max) + 1
    ):
        raise ValueError("manifest has an invalid next_vector_id")

    vector_ids = _database_vector_ids(database_path)
    if vector_ids and max(vector_ids) >= next_vector_id:
        raise ValueError("manifest next_vector_id is not greater than existing ids")
    return next_vector_id


def _database_vector_ids(path: Path) -> list[int]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT vector_id FROM chunks").fetchall()

    vector_ids = []
    for (value,) in rows:
        if not isinstance(value, str) or not value.isdecimal():
            raise ValueError(f"invalid chunks.vector_id: {value!r}")
        vector_id = int(value)
        if vector_id > np.iinfo(np.int64).max:
            raise ValueError(f"chunks.vector_id is outside int64 range: {value}")
        vector_ids.append(vector_id)

    if len(vector_ids) != len(set(vector_ids)):
        raise ValueError("chunks.vector_id values must be unique")
    return vector_ids


def _faiss_vector_ids(index: Any) -> list[int]:
    if not isinstance(index, faiss.IndexIDMap2):
        raise TypeError("FAISS index must be IndexIDMap2")
    return [int(value) for value in faiss.vector_to_array(index.id_map)]


def _normalize_vector_ids(vector_ids: Sequence[int | str]) -> list[int]:
    if isinstance(vector_ids, (str, bytes)):
        raise TypeError("vector_ids must be a sequence of integers")

    result = []
    for value in vector_ids:
        if isinstance(value, bool):
            raise TypeError(f"invalid vector_id: {value!r}")
        if isinstance(value, int):
            vector_id = value
        elif isinstance(value, str) and value.isdecimal():
            vector_id = int(value)
        else:
            raise TypeError(f"invalid vector_id: {value!r}")
        if vector_id < 0 or vector_id > np.iinfo(np.int64).max:
            raise ValueError(f"vector_id is outside int64 range: {value!r}")
        result.append(vector_id)

    if len(result) != len(set(result)):
        raise ValueError("vector_ids contain duplicates")
    return result


def _copy_database(source: Path, destination: Path) -> None:
    with (
        sqlite3.connect(source) as source_connection,
        sqlite3.connect(destination) as destination_connection,
    ):
        source_connection.backup(destination_connection)


def _delete_database_vectors(path: Path, vector_ids: Sequence[int]) -> int:
    placeholders = ", ".join("?" for _ in vector_ids)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            f"DELETE FROM chunks WHERE vector_id IN ({placeholders})",
            tuple(str(value) for value in vector_ids),
        )
        return cursor.rowcount


def _prepare_image(
    source: Any,
    chunk_index: int,
) -> tuple[ImageAsset, str, str]:
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
    extension = IMAGE_EXTENSIONS.get(mime_type) or mimetypes.guess_extension(mime_type)
    if extension is None:
        raise ValueError(
            f"image chunk at index {chunk_index} has unsupported "
            f"MIME type {mime_type!r}"
        )
    if extension == ".jpe":
        extension = ".jpg"

    filename = f"{image_sha256[:16]}{extension}"
    relative_path = (Path(IMAGES_DIRECTORY) / filename).as_posix()
    return ImageAsset(relative_path, content), mime_type, image_sha256


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


def _write_images(directory: Path, image_assets: Sequence[ImageAsset]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for image in image_assets:
        (directory / Path(image.relative_path).name).write_bytes(image.content)


def _document_from_row(row: sqlite3.Row, index: int) -> Document:
    document_id = _uuid_value(row["id"], f"document at index {index} has invalid id")
    file_sha = row["file_sha256"]
    if (
        not isinstance(file_sha, str)
        or len(file_sha) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in file_sha)
    ):
        raise ValueError(f"document {document_id} has invalid file_sha256")

    values = {}
    for key in ("file_name", "updated_at", "status"):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"document {document_id} has invalid {key}")
        values[key] = value.strip()

    return Document(
        id=document_id,
        file_sha256=file_sha.lower(),
        file_name=values["file_name"],
        updated_at=values["updated_at"],
        status=values["status"],
    )


def _required_uuid(
    chunk: Mapping[str, Any],
    keys: Sequence[str],
    index: int,
) -> str:
    for key in keys:
        value = chunk.get(key)
        if value is not None:
            return _uuid_value(value, f"chunk at index {index} has an invalid {key}")
    raise ValueError(f"chunk at index {index} has no {keys[0]}")


def _uuid_value(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    try:
        return str(UUID(value.strip()))
    except ValueError as error:
        raise ValueError(message) from error


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "COMPLETED_STATUS",
    "Document",
    "PENDING_STATUS",
    "delete_vectors",
    "file_sha256",
    "initialize_database",
    "persist_chunks",
    "read_pending_documents",
]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .chunking import chunk_blocks
from .embedding import EMBEDDING_MODEL, embed_chunks
from .normalizer import normalize_blocks
from .parsers import DocxParser, ParserRegistry
from .persisting import (
    DATA_DIRECTORY,
    Document,
    database_chunk_count,
    file_sha256,
    persist_chunks,
    read_pending_documents,
)

SOURCE_DIRECTORY = Path("source")


def build_index(
    *,
    data_directory: str | Path = DATA_DIRECTORY,
    source_directory: str | Path = SOURCE_DIRECTORY,
) -> dict[str, Any] | None:
    documents = read_pending_documents(data_directory)
    if not documents:
        print("No pending documents.")
        return None

    if database_chunk_count(data_directory):
        raise RuntimeError(
            "chunks already exist; incremental index updates are not implemented"
        )

    parser_registry = ParserRegistry()
    parser_registry.register(DocxParser)
    embedded_chunks: list[dict[str, Any]] = []

    for document in documents:
        file = _document_path(source_directory, document)
        actual_sha256 = file_sha256(file)
        if actual_sha256 != document.file_sha256:
            raise ValueError(
                f"SHA-256 mismatch for document {document.id}: {document.file_name}"
            )

        result = parser_registry.parse(file)
        data = _parser_output(result, document)
        blocks = normalize_blocks(data)
        chunks = chunk_blocks(blocks, document_id=document.id)
        if not chunks:
            raise ValueError(f"document {document.id} produced no chunks")
        embedded_chunks.extend(embed_chunks(chunks))

    manifest = persist_chunks(
        embedded_chunks,
        documents=documents,
        embedding_model=EMBEDDING_MODEL,
        data_directory=data_directory,
    )
    print(
        f"Persisted {manifest['chunk_count']} chunks to "
        f"{Path(data_directory) / manifest['files']['sqlite']} and "
        f"{Path(data_directory) / manifest['files']['faiss']}"
    )
    return manifest


def _document_path(source_directory: str | Path, document: Document) -> Path:
    source_root = Path(source_directory).resolve()
    file = (source_root / document.file_name).resolve()
    try:
        file.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"document {document.id} file_name escapes the source directory"
        ) from error
    if not file.is_file():
        raise FileNotFoundError(file)
    return file


def _parser_output(
    result: Any,
    document: Document,
) -> Sequence[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise TypeError(f"parser returned an invalid result for document {document.id}")
    output = result.get("output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise TypeError(f"parser returned invalid output for document {document.id}")
    if not output:
        raise ValueError(f"parser returned no output for document {document.id}")
    if any(not isinstance(block, Mapping) for block in output):
        raise TypeError(f"parser returned an invalid block for document {document.id}")
    return output


__all__ = ["build_index"]

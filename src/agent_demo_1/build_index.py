import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .chunking import chunk_blocks
from .embedding import EMBEDDING_MODEL, embed_chunks
from .normalizer import normalize_blocks
from .parsers import DocxParser, ParserRegistry
from .persisting import (
    DATA_DIRECTORY,
    Document,
    mark_documents_failed,
    mark_documents_processing,
    persist_chunks,
)

SOURCE_DIRECTORY = Path("source")
DATA_SOURCE_DIRECTORY = DATA_DIRECTORY / "source"


def build_index(document_ids: Sequence[str]):
    documents = mark_documents_processing(document_ids)
    try:
        parser_registry = ParserRegistry()
        parser_registry.register(DocxParser)
        embedded_chunks: list[dict[str, Any]] = []

        for document in documents:
            file = _document_path(document)
            result = parser_registry.parse(file)
            blocks = normalize_blocks(result.output)
            chunks = chunk_blocks(blocks, document_id=document.id)
            if not chunks:
                raise ValueError(f"document {document.id} produced no chunks")
            embedded_chunks.extend(embed_chunks(chunks))

        moved_files = _archive_documents(documents)
        try:
            persist_chunks(
                embedded_chunks,
                documents=documents,
                embedding_model=EMBEDDING_MODEL,
            )
        except Exception:
            _restore_documents(moved_files)
            raise
    except Exception:
        mark_documents_failed(document_ids)
        raise


def _document_path(document: Document) -> Path:
    source_root = SOURCE_DIRECTORY.resolve()
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


def _archive_documents(documents: list[Document]) -> list[tuple[Path, Path]]:
    DATA_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_root = SOURCE_DIRECTORY.resolve()
    archived_root = DATA_SOURCE_DIRECTORY.resolve()
    moved_files: list[tuple[Path, Path]] = []
    try:
        for document in documents:
            source = (source_root / document.file_name).resolve()
            destination = (archived_root / document.file_name).resolve()
            try:
                source.relative_to(source_root)
                destination.relative_to(archived_root)
            except ValueError as error:
                raise ValueError(
                    f"document {document.id} file_name escapes a source directory"
                ) from error
            if not source.is_file():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved_files.append((source, destination))
    except Exception:
        _restore_documents(moved_files)
        raise
    return moved_files


def _restore_documents(moved_files: list[tuple[Path, Path]]) -> None:
    for source, destination in reversed(moved_files):
        if destination.exists():
            os.replace(destination, source)


__all__ = ["build_index"]

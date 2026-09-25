import os
from collections.abc import Sequence
from pathlib import Path

from .chunking import chunk_blocks
from .embedding import EMBEDDING_MODEL, embed_chunks
from .normalizer import normalize_blocks
from .parsers import DocxParser, ParserRegistry, PdfParser
from .persisting import (
    DATA_DIRECTORY,
    Document,
    mark_documents_failed,
    mark_documents_processing,
    persist_chunks,
)

SOURCE_DIRECTORY = Path("source")
DATA_SOURCE_DIRECTORY = DATA_DIRECTORY / "source"


def build_index(document_ids: Sequence[str]) -> None:
    process_documents(mark_documents_processing(document_ids))


def process_documents(documents: Sequence[Document]) -> None:
    parser_registry = ParserRegistry()
    parser_registry.register(DocxParser)
    parser_registry.register(PdfParser)
    errors: list[Exception] = []

    for document in documents:
        try:
            file = _document_path(document)
            result = parser_registry.parse(file)
            blocks = normalize_blocks(result["output"])
            chunks = chunk_blocks(blocks, document_id=document.id)
            if not chunks:
                raise ValueError(f"document {document.id} produced no chunks")
            embedded_chunks = embed_chunks(chunks)

            moved_files = _archive_documents([document])
            try:
                persist_chunks(
                    embedded_chunks,
                    documents=[document],
                    embedding_model=EMBEDDING_MODEL,
                )
            except Exception:
                _restore_documents(moved_files)
                raise
        except Exception as error:
            mark_documents_failed([document.id])
            error.add_note(f"document {document.id}: {document.file_name}")
            errors.append(error)

    if errors:
        raise ExceptionGroup("document processing failed", errors)


def _document_path(document: Document) -> Path:
    for directory in (SOURCE_DIRECTORY, DATA_SOURCE_DIRECTORY):
        source_root = directory.resolve()
        file = (source_root / document.file_name).resolve()
        try:
            file.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"document {document.id} file_name escapes the source directory"
            ) from error
        if file.is_file():
            return file
    raise FileNotFoundError(SOURCE_DIRECTORY / document.file_name)


def _archive_documents(documents: Sequence[Document]) -> list[tuple[Path, Path]]:
    DATA_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archived_root = DATA_SOURCE_DIRECTORY.resolve()
    moved_files: list[tuple[Path, Path]] = []
    try:
        for document in documents:
            source = _document_path(document)
            destination = (archived_root / document.file_name).resolve()
            try:
                destination.relative_to(archived_root)
            except ValueError as error:
                raise ValueError(
                    f"document {document.id} file_name escapes a source directory"
                ) from error
            if source == destination:
                continue
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


__all__ = ["build_index", "process_documents"]

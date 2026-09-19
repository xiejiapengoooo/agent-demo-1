from pathlib import Path
from typing import Any

from .chunking import chunk_blocks
from .embedding import EMBEDDING_MODEL, embed_chunks
from .normalizer import normalize_blocks
from .parsers import DocxParser, ParserRegistry
from .persisting import (
    Document,
    persist_chunks,
    read_pending_documents,
)

SOURCE_DIRECTORY = Path("source")


def build_index():
    documents = read_pending_documents()
    if not documents:
        print("No pending documents.")
        return

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

    manifest = persist_chunks(
        embedded_chunks,
        documents=documents,
        embedding_model=EMBEDDING_MODEL,
    )
    print(manifest)


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


__all__ = ["build_index"]

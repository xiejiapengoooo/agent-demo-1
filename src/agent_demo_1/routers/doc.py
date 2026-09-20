import hashlib
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ..build_index import build_index
from ..persisting import (
    DATA_DIRECTORY,
    WAITING_STATUS,
    Document,
    DocumentAlreadyExistsError,
    read_documents,
    read_waiting_documents,
    register_document,
)
from ..persisting import (
    delete_document as delete_document_record,
)

router = APIRouter()
SOURCE_DIRECTORY = Path("source")
DATA_SOURCE_DIRECTORY = DATA_DIRECTORY / "source"
UPLOAD_CHUNK_SIZE = 1024 * 1024
DOCUMENT_UPLOAD_FILE = File(...)


@router.post("/document")
async def post_document(file: UploadFile = DOCUMENT_UPLOAD_FILE) -> Response:
    raw_filename = (file.filename or "").replace("\\", "/")
    filename = Path(raw_filename).name
    if not filename or filename in {".", ".."} or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded document must have a filename",
        )

    destination_directory = SOURCE_DIRECTORY
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / filename
    temporary_path: Path | None = None

    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            suffix=".tmp",
            dir=destination_directory,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                digest.update(chunk)
                output.write(chunk)

        try:
            register_document(
                file_name=filename, file_digest=digest.hexdigest().lower()
            )
        except DocumentAlreadyExistsError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="document already exists",
            ) from error

        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.delete("/document", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
) -> Response:
    document = next(
        (item for item in read_documents() if item.id == document_id),
        None,
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )

    source_paths = []
    for source_root in (SOURCE_DIRECTORY, DATA_SOURCE_DIRECTORY):
        resolved_root = source_root.resolve()
        source_path = (resolved_root / document.file_name).resolve()
        try:
            source_path.relative_to(resolved_root)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="document path escapes the source directory",
            ) from error
        if source_path.exists() and not source_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="document source is not a file",
            )
        source_paths.append(source_path)

    deleted = delete_document_record(document.id)
    if deleted is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )

    for source_path in source_paths:
        source_path.unlink(missing_ok=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/documents")
async def list_documents() -> list[Document]:
    return read_documents()


@router.post("/documents/start", status_code=status.HTTP_202_ACCEPTED)
async def start_document_processing(
    background_tasks: BackgroundTasks,
    document_ids: Annotated[list[str] | None, Body(embed=True, min_length=1)] = None,
) -> Response:
    documents = read_waiting_documents() if document_ids is None else read_documents()
    ids = (
        [document.id for document in documents]
        if document_ids is None
        else list(dict.fromkeys(document_ids))
    )
    documents_by_id = {document.id: document for document in documents}
    missing_ids = [
        document_id for document_id in ids if document_id not in documents_by_id
    ]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"documents not found: {', '.join(missing_ids)}",
        )

    not_waiting = [
        document_id
        for document_id in ids
        if documents_by_id[document_id].status != WAITING_STATUS
    ]
    if not_waiting:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"documents are not waiting: {', '.join(not_waiting)}",
        )

    if ids:
        background_tasks.add_task(build_index, ids)
    return Response(status_code=status.HTTP_202_ACCEPTED)


__all__ = [
    "router",
]

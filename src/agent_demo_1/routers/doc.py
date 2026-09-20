import hashlib
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from ..persisting import (
    Document,
    DocumentAlreadyExistsError,
    read_documents,
    register_document,
)

router = APIRouter()
SOURCE_DIRECTORY = Path("source")
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


@router.get("/documents")
async def list_documents() -> list[Document]:
    return read_documents()


__all__ = [
    "router",
]

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

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

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                output.write(chunk)
    finally:
        await file.close()

    return Response(status_code=status.HTTP_202_ACCEPTED)


__all__ = [
    "router",
]

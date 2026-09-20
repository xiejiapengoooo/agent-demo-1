from fastapi import APIRouter, Response, status

router = APIRouter()


@router.post("/document")
async def get_results():
    return Response(status_code=status.HTTP_200_OK)


__all__ = [
    "router",
]

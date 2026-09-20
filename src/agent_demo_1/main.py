import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import doc_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(doc_router)

    return app


def main() -> None:
    settings = get_settings()

    uvicorn.run(
        create_app(),
        host=settings.host,
        port=settings.port,
        reload=False,
    )


__all__ = ["main"]

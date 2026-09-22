from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .persisting import recover_processing_documents
from .routers import chat_router, doc_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator:
    recover_processing_documents()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(doc_router)
    app.mount(
        "/",
        StaticFiles(directory=Path("static"), html=True),
    )

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

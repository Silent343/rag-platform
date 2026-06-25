"""
FastAPI application entry point for the RAG backend.

Wires up CORS (so the Angular dashboard can call the API) and mounts the
routes. Run locally with:

    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings


def create_app() -> FastAPI:
    """
    Builds and configures the FastAPI application.

    Returns:
        The configured FastAPI instance.
    """
    settings = get_settings()
    app = FastAPI(
        title="Document RAG API",
        description="Ask questions over your own documents, answered with citations.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        """
        Simple liveness probe.

        Returns:
            A small status payload.
        """
        return {"status": "ok"}

    return app


app = create_app()

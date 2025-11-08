"""
FastAPI server for Mindcore framework.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import ingest_router, context_router
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Mindcore API",
        description="Intelligent memory and context management for AI agents",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(ingest_router)
    app.include_router(context_router)

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Mindcore API",
            "version": "0.1.0",
            "description": "Intelligent memory and context management for AI agents",
            "endpoints": {
                "docs": "/docs",
                "ingest": "/ingest",
                "context": "/context",
                "health": "/health"
            }
        }

    # Health check endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        from .. import get_mindcore_instance

        try:
            mindcore = get_mindcore_instance()
            cache_stats = mindcore.cache.get_stats()

            return {
                "status": "healthy",
                "cache": cache_stats,
                "database": "connected"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

    logger.info("FastAPI application created")
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        debug: Enable debug mode.
    """
    app = create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info"
    )


if __name__ == "__main__":
    run_server()

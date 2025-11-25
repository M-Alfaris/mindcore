"""
FastAPI server for Mindcore framework.
"""
import os
from typing import List
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import ingest_router, context_router
from ..utils.logger import get_logger
from ..utils.security import get_rate_limiter, RateLimiter

logger = get_logger(__name__)


def get_cors_origins() -> List[str]:
    """
    Get CORS allowed origins from environment or config.

    Returns:
        List of allowed origins.
    """
    origins_env = os.getenv("MINDCORE_CORS_ORIGINS", "")
    if origins_env:
        return [origin.strip() for origin in origins_env.split(",") if origin.strip()]

    # Default: allow localhost for development
    return [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]


def create_app(cors_origins: List[str] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        cors_origins: Optional list of allowed CORS origins. If None, uses
                     MINDCORE_CORS_ORIGINS env var or localhost defaults.

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

    # CORS middleware - properly configured
    allowed_origins = cors_origins or get_cors_origins()

    # Note: allow_credentials=True requires specific origins, not "*"
    # If you need wildcard origins, set allow_credentials=False
    if "*" in allowed_origins:
        # Wildcard mode: no credentials
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    else:
        # Specific origins: credentials allowed
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

    # Health check endpoint - lightweight, doesn't auto-initialize Mindcore
    @app.get("/health")
    async def health():
        """
        Health check endpoint.

        Returns basic health status without initializing Mindcore.
        Use /health/full for detailed status including database connectivity.
        """
        return {
            "status": "healthy",
            "service": "mindcore-api",
            "version": "0.1.0"
        }

    @app.get("/health/full")
    async def health_full():
        """
        Full health check including database and cache status.

        This endpoint will initialize Mindcore if not already initialized.
        """
        from .. import _mindcore_instance

        # Check if Mindcore is initialized without forcing initialization
        if _mindcore_instance is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_initialized",
                    "message": "Mindcore not yet initialized. Make a request to /ingest or /context first."
                }
            )

        try:
            cache_stats = _mindcore_instance.cache.get_stats()

            # Test database connectivity
            db_status = "connected"
            try:
                with _mindcore_instance.db.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
            except Exception as db_err:
                db_status = f"error: {type(db_err).__name__}"

            return {
                "status": "healthy",
                "cache": cache_stats,
                "database": db_status
            }
        except Exception as e:
            logger.error(f"Full health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": type(e).__name__}
            )

    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        logger.info("Mindcore API starting up...")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources on shutdown."""
        from .. import _mindcore_instance

        logger.info("Mindcore API shutting down...")
        if _mindcore_instance is not None:
            try:
                _mindcore_instance.close()
                logger.info("Mindcore client closed successfully")
            except Exception as e:
                logger.error(f"Error closing Mindcore client: {e}")

        # Cleanup rate limiter
        rate_limiter = get_rate_limiter()
        rate_limiter.cleanup_stale_entries()

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
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

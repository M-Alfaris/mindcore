"""
FastAPI server for Mindcore framework.

Run with:
    mindcore-server                    # CLI command
    python -m mindcore.api.server      # Module
    uvicorn mindcore.api.server:app    # ASGI server

Configuration via environment variables:
    OPENAI_API_KEY          - Required for OpenAI LLM
    MINDCORE_DB_HOST        - Database host (default: localhost)
    MINDCORE_DB_PORT        - Database port (default: 5432)
    MINDCORE_DB_NAME        - Database name (default: mindcore)
    MINDCORE_DB_USER        - Database user (default: postgres)
    MINDCORE_DB_PASSWORD    - Database password
    MINDCORE_DB_USE_PGBOUNCER - Set to 'true' for PgBouncer mode
    MINDCORE_USE_SQLITE     - Set to 'true' to use SQLite instead
    MINDCORE_SQLITE_PATH    - SQLite database path (default: mindcore.db)
    MINDCORE_CORS_ORIGINS   - Comma-separated CORS origins
    MINDCORE_TIMEZONE       - IANA timezone (default: UTC)
    MINDCORE_LOG_LEVEL      - Log level (default: INFO)
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import ingest_router, context_router, dashboard_router
from ..utils.logger import get_logger
from ..utils.security import get_rate_limiter, RateLimiter

logger = get_logger(__name__)

# Global Mindcore instance (initialized on startup)
_mindcore_client = None


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


def get_mindcore_client():
    """
    Get or create the Mindcore client instance.

    Reads configuration from environment variables.
    """
    global _mindcore_client

    if _mindcore_client is not None:
        return _mindcore_client

    from .. import MindcoreClient

    # Determine database mode
    use_sqlite = os.getenv("MINDCORE_USE_SQLITE", "").lower() == "true"
    sqlite_path = os.getenv("MINDCORE_SQLITE_PATH", "mindcore.db")

    # LLM provider
    llm_provider = os.getenv("MINDCORE_LLM_PROVIDER", "auto")

    logger.info(f"Initializing Mindcore client " f"(sqlite={use_sqlite}, llm={llm_provider})")

    _mindcore_client = MindcoreClient(
        use_sqlite=use_sqlite,
        sqlite_path=sqlite_path,
        llm_provider=llm_provider,
        persistent_cache=True,
    )

    return _mindcore_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown.

    Initializes Mindcore on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Mindcore API starting up...")

    # Pre-initialize Mindcore client (optional - can also lazy init on first request)
    try:
        client = get_mindcore_client()
        logger.info(f"Mindcore initialized with {client.provider_name} LLM provider")
    except Exception as e:
        logger.warning(f"Mindcore pre-initialization failed: {e}")
        logger.info("Mindcore will initialize on first request")

    yield

    # Shutdown
    logger.info("Mindcore API shutting down...")
    global _mindcore_client
    if _mindcore_client is not None:
        try:
            _mindcore_client.close()
            logger.info("Mindcore client closed successfully")
        except Exception as e:
            logger.error(f"Error closing Mindcore client: {e}")
        _mindcore_client = None


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
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware - properly configured
    allowed_origins = cors_origins or get_cors_origins()

    # Note: allow_credentials=True requires specific origins, not "*"
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
    app.include_router(dashboard_router)

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Mindcore API",
            "version": "0.2.0",
            "description": "Intelligent memory and context management for AI agents",
            "endpoints": {
                "docs": "/docs",
                "ingest": "/ingest",
                "context": "/context",
                "health": "/health",
                "health_full": "/health/full",
                "dashboard": "/api/dashboard",
            },
            "status": "running",
        }

    # Health check endpoint - lightweight, no initialization
    @app.get("/health")
    async def health():
        """
        Basic health check endpoint.

        Returns service status without initializing Mindcore.
        Use /health/full for detailed status including database.
        """
        return {"status": "healthy", "service": "mindcore-api", "version": "0.2.0"}

    @app.get("/health/full")
    async def health_full():
        """
        Full health check including database, cache, and LLM status.

        This endpoint initializes Mindcore if not already done.
        """
        global _mindcore_client

        result = {
            "status": "healthy",
            "service": "mindcore-api",
            "version": "0.2.0",
            "components": {},
        }

        # Check Mindcore initialization
        if _mindcore_client is None:
            try:
                _mindcore_client = get_mindcore_client()
            except Exception as e:
                result["status"] = "unhealthy"
                result["components"]["mindcore"] = {"status": "error", "error": str(e)}
                return JSONResponse(status_code=503, content=result)

        # Check database
        try:
            if hasattr(_mindcore_client.db, "health_check"):
                db_health = _mindcore_client.db.health_check()
            else:
                # Fallback for SQLite or older db managers
                with _mindcore_client.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                db_health = {"status": "healthy"}

            result["components"]["database"] = db_health
        except Exception as e:
            result["status"] = "degraded"
            result["components"]["database"] = {"status": "error", "error": str(e)}

        # Check cache
        try:
            cache_stats = _mindcore_client.cache.get_stats()
            result["components"]["cache"] = {"status": "healthy", **cache_stats}
        except Exception as e:
            result["components"]["cache"] = {"status": "error", "error": str(e)}

        # Check LLM provider
        try:
            result["components"]["llm"] = {
                "status": "healthy",
                "provider": _mindcore_client.provider_name,
            }
        except Exception as e:
            result["components"]["llm"] = {"status": "error", "error": str(e)}

        # Overall status
        if any(c.get("status") == "error" for c in result["components"].values()):
            if result["status"] == "healthy":
                result["status"] = "degraded"

        status_code = 200 if result["status"] == "healthy" else 503
        return JSONResponse(status_code=status_code, content=result)

    @app.get("/ready")
    async def readiness():
        """
        Kubernetes-style readiness probe.

        Returns 200 if the service is ready to accept traffic.
        """
        global _mindcore_client

        if _mindcore_client is None:
            return JSONResponse(
                status_code=503, content={"ready": False, "reason": "not_initialized"}
            )

        return {"ready": True}

    @app.get("/live")
    async def liveness():
        """
        Kubernetes-style liveness probe.

        Returns 200 if the service is alive.
        """
        return {"alive": True}

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": "Internal server error", "type": type(exc).__name__}
        )

    logger.info("FastAPI application created")
    return app


def run_server(host: str = None, port: int = None, debug: bool = False, workers: int = None):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to (default: 0.0.0.0 or MINDCORE_HOST)
        port: Port to bind to (default: 8000 or MINDCORE_PORT)
        debug: Enable debug mode
        workers: Number of worker processes (default: 1)
    """
    host = host or os.getenv("MINDCORE_HOST", "0.0.0.0")
    port = port or int(os.getenv("MINDCORE_PORT", "8000"))
    workers = workers or int(os.getenv("MINDCORE_WORKERS", "1"))
    log_level = os.getenv("MINDCORE_LOG_LEVEL", "info").lower()

    if debug:
        log_level = "debug"

    logger.info(f"Starting Mindcore API server on {host}:{port}")

    uvicorn.run(
        "mindcore.api.server:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers if not debug else 1,
        reload=debug,
    )


# Default app instance for ASGI servers (e.g., uvicorn mindcore.api.server:app)
app = create_app()


if __name__ == "__main__":
    run_server()

"""API routes for Mindcore framework."""

from .context import router as context_router
from .dashboard import router as dashboard_router
from .ingest import router as ingest_router


__all__ = ["context_router", "dashboard_router", "ingest_router"]

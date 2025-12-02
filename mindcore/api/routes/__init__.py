"""
API routes for Mindcore framework.
"""

from .ingest import router as ingest_router
from .context import router as context_router
from .dashboard import router as dashboard_router

__all__ = ["ingest_router", "context_router", "dashboard_router"]
